import networkx as nx
import torch
import numpy as np
import pandas as pd

from rl4co.utils.ops import gather_by_index, get_tour_length
from rl4co.envs import CVRPEnv
from rl4co.envs.routing.mtvrp.env import MTVRPEnv
from rl4co.envs.routing.mtvrp.generator import MTVRPGenerator
from rl4co.models.zoo.am import AttentionModel
from rl4co.models import AttentionModelPolicy, REINFORCE
from rl4co.utils.trainer import RL4COTrainer
from tensordict import TensorDict
from matplotlib import pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"]='TRUE'

## 在这里覆写td_init中的locs，demand等信息
data_Coord = pd.read_excel('HKY_WT.xlsx', header=None)
# data_Coord = pd.read_excel('154WTs.xls', header=None)
start_index_WT = 2
stop_index_WT = 43
start_index_OS = 44
stop_index_OS = 44
x_col_WT = 2
y_col_WT = 3

index_WT = np.arange(start_index_WT - 1, stop_index_WT, 1)
Coord_WT = data_Coord.loc[index_WT, [x_col_WT - 1, y_col_WT - 1]].values
Coord_WT = np.array(Coord_WT)
index_OS = np.arange(start_index_OS - 1, stop_index_OS, 1)
Coord_OS = data_Coord.loc[index_OS, [x_col_WT - 1, y_col_WT - 1]].values
Coord_OS = np.array(Coord_OS)
WTs = stop_index_WT - start_index_WT + 1
OSs = stop_index_OS - start_index_OS + 1

Coords = np.append(Coord_OS, Coord_WT, axis=0)
Coords_x_max = Coords[:, 0].max()
Coords_x_min = Coords[:, 0].min()
Coords_y_max = Coords[:, 1].max()
Coords_y_min = Coords[:, 1].min()
Coords_x_means = (Coords_x_max + Coords_x_min) / 2
Coords_y_means = (Coords_y_max + Coords_y_min) / 2
Coordrs_scale = max(Coords_x_max - Coords_x_min, Coords_y_max - Coords_y_min)
Coords = ( (Coords - [Coords_x_means, Coords_y_means]) / [Coordrs_scale, Coordrs_scale] ) + [0.5,0.5]
# Coords = (Coords - [Coords_x_min, Coords_y_min]) / [Coordrs_scale, Coordrs_scale]
Coords = Coords.astype(np.float32)
coords_tensordict = TensorDict({'locs': torch.tensor(Coords)}, batch_size=[WTs+OSs, 2])
coords_tensordict = coords_tensordict.unsqueeze(0)


# RL4CO env based on TorchRL
generator_params={'num_loc': WTs,'loc_distribution': "uniform", 'specified_demand': 1/6, 'specified_capacity': 1+1/6*WTs, 'check_solution':True,'specified_locs':coords_tensordict['locs'],'specified':True}
# env = CVRPEnv(generator_params={'num_loc': WTs,'check_solution':True})
env = CVRPEnv(generator_params=generator_params)

# Policy: neural network, in this case with encoder-decoder architecture
policy = AttentionModelPolicy(env_name=env.name,
                              embed_dim=128,
                              num_encoder_layers=3,
                              num_heads=8,
                            )

# Greedy rollouts over untrained policy
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

td_init = env.reset(batch_size=[1]).to(device)
td_init['locs'] = coords_tensordict['locs']
td_init['demand'] = torch.full(td_init['demand'].shape, 1/6, device=td_init.device)
td_init['capacity'] = torch.full(td_init['capacity'].shape, 1+1/6*WTs, device=td_init.device)


policy = policy.to(device)
# out = policy(td_init.clone(), phase="test", decode_type="greedy", return_actions=True)
out = policy(td_init.clone(), phase="test", decode_type="greedy", return_actions=True)
actions_untrained = out['actions'].cpu().detach()
rewards_untrained = out['reward'].cpu().detach()

# for i in range(3):
#     print(f"Problem {i+1} | Cost: {-rewards_untrained[i]:.3f}")
#     env.render(td_init[i], actions_untrained[i])

trainer = RL4COTrainer(
    max_epochs=100,
    accelerator="gpu",
    devices=1,
    logger=None,
)

# RL Model: REINFORCE and greedy rollout baseline
model = REINFORCE(env,
                    policy,
                    baseline="critic",
                    batch_size=512,
                    train_data_size=50_000,
                    val_data_size=5_000,
                    optimizer_kwargs={"lr": 1e-4},
                    )

# trainer.fit(model)
model = AttentionModel.load_from_checkpoint("test_basic_100epochs.ckpt", load_baseline=False)

# get trained policy
policy = model.policy.to(device)
out = policy(td_init.clone(), phase="test", decode_type="greedy", return_actions=True)
# out = policy(td_init.clone(), phase="test", decode_type="beam_search", return_actions=True)
# out = policy(td_init.clone(), phase="test", decode_type="sampling", return_actions=True)
actions_trained = out['actions'].cpu().detach()
rewards_trained = out['reward'].cpu().detach()

##保存与加载
# checkpoint_path = "test_ovrp_modified_154WT_25epoch_50000datasize.ckpt"
# trainer.save_checkpoint(checkpoint_path)
##计算成本
actions_trained2 = actions_trained.to(device)
locs_ordered = torch.cat([td_init["locs"][..., 0:1, :], gather_by_index(td_init["locs"], actions_trained.to(device)),],dim=1,)
locs_ordered_next = torch.roll(locs_ordered, -1, dims=-2)
cable_length = (locs_ordered_next - locs_ordered).norm(p=2, dim=-1)
is_WT_tensor = (actions_trained2 != 0).clone().detach().float().to(actions_trained2.device)
WT_power = is_WT_tensor.clone()
cable_power = WT_power.clone()
cable_cost = WT_power.clone()
for i in range(is_WT_tensor.size(1) - 2, -1, -1):
    cable_power[:, i] = is_WT_tensor[:, i] * (cable_power[:, i] + cable_power[:, i + 1])
# ##获取每段电缆成本
cable_cost = torch.where(cable_power == 1, 1.854, cable_cost)
cable_cost = torch.where(cable_power == 2, 1.854, cable_cost)
cable_cost = torch.where(cable_power == 3, 2.292, cable_cost)
cable_cost = torch.where(cable_power == 4, 2.803, cable_cost)
cable_cost = torch.where(cable_power == 5, 3.555, cable_cost)
cable_cost = torch.where(cable_power == 6, 4.827, cable_cost)
cable_cost = torch.where(cable_power >= 7, 8000, cable_cost)
cable_cost = torch.cat([cable_cost.clone(),
                    torch.zeros(*actions_trained2.shape[:-1], 1, device=is_WT_tensor.device),],
                dim=1,)
cable_cost_total = (cable_cost*cable_length).sum(-1)


# Plotting
node_num = WTs + OSs
nodes = list(range(node_num))
node_Coord = np.append(Coord_OS, Coord_WT, axis=0)
node_withCoord = dict(zip(nodes, node_Coord))
node_B = actions_trained.numpy()
node_B = np.append(node_B, 0)
node_A = np.roll(node_B, 1, axis=-1)

G_result = nx.Graph()
G_result.add_nodes_from(nodes)
# edge_index = np.sum(x_value, axis=1)
for i in range(node_B.shape[0]):
    if node_B[i] != 0:
        G_result.add_edge(node_A[i], node_B[i])
nx.draw(G_result, pos=node_withCoord, node_size=10, node_color='r')  # 绘制节点
x_max, y_max = node_Coord.max(axis=0)  # 获取每一列最大值
x_min, y_min = node_Coord.min(axis=0)  # 获取每一列最小值
x_num = (x_max - x_min) / 10
y_num = (y_max - y_min) / 10
plt.xlim(x_min - x_num, x_max + x_num)
plt.ylim(y_min - y_num, y_max + y_num)
plt.show()  # 显示图像
print('*********  线路规划建设成本：￥ %f *********' % (1e4 * Coordrs_scale*cable_cost_total))
print('运行结束!')
# import matplotlib.pyplot as plt
# for i, td in enumerate(td_init):
#     fig, axs = plt.subplots(1,2, figsize=(11,5))
#     env.render(td, actions_untrained[i], ax=axs[0])
#     env.render(td, actions_trained[i], ax=axs[1])
#     axs[0].set_title(f"Untrained | Cost = {-rewards_untrained[i].item():.3f}")
#     axs[1].set_title(r"Trained $\pi_\theta$" + f"| Cost = {-out['reward'][i].item():.3f}")



