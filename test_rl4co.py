import torch
import numpy as np
import pandas as pd

from rl4co.envs import TSPEnv
from rl4co.envs import CVRPEnv
from rl4co.models import AttentionModelPolicy, REINFORCE
from rl4co.utils.trainer import RL4COTrainer

# RL4CO env based on TorchRL
# env = TSPEnv(generator_params={'num_loc': 50})
env = CVRPEnv(generator_params={'num_loc': 50})

# Policy: neural network, in this case with encoder-decoder architecture
policy = AttentionModelPolicy(env_name=env.name,
                              embed_dim=128,
                              num_encoder_layers=3,
                              num_heads=8,
                            )

# RL Model: REINFORCE and greedy rollout baseline
model = REINFORCE(env,
                    policy,
                    baseline="rollout",
                    batch_size=512,
                    train_data_size=100_000,
                    val_data_size=10_000,
                    optimizer_kwargs={"lr": 1e-4},
                    )

# Greedy rollouts over untrained policy
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
td_init = env.reset(batch_size=[3]).to(device)
## 在这里覆写td_init中的locs，demand等信息
data_Coord = pd.read_excel('HKY_WT.xlsx', header=None)
# data = torch.tensor(data_Coord.values)
# start_index_WT = 2
# stop_index_WT = 43
# start_index_OS = 44
# stop_index_OS = 44
# x_col_WT = 2
# y_col_WT = 3
#
# index_WT = np.arange(start_index_WT - 1, stop_index_WT, 1)
# Coord_WT = data_Coord.loc[index_WT, [x_col_WT - 1, y_col_WT - 1]].values
# Coord_WT = np.array(Coord_WT)
# index_OS = np.arange(start_index_OS - 1, stop_index_OS, 1)
# Coord_OS = data_Coord.loc[index_OS, [x_col_WT - 1, y_col_WT - 1]].values
# Coord_OS = np.array(Coord_OS)
# WTs = stop_index_WT - start_index_WT + 1
# OSs = stop_index_OS - start_index_OS + 1


policy = policy.to(device)
out = policy(td_init.clone(), phase="test", decode_type="greedy", return_actions=True)
actions_untrained = out['actions'].cpu().detach()
rewards_untrained = out['reward'].cpu().detach()

# for i in range(3):
#     print(f"Problem {i+1} | Cost: {-rewards_untrained[i]:.3f}")
#     env.render(td_init[i], actions_untrained[i])

trainer = RL4COTrainer(
    max_epochs=3,
    accelerator="gpu",
    devices=1,
    logger=None,
)

trainer.fit(model)

# Greedy rollouts over trained model (same states as previous plot)
policy = model.policy.to(device)
out = policy(td_init.clone(), phase="test", decode_type="greedy", return_actions=True)
actions_trained = out['actions'].cpu().detach()

# # Plotting
# import matplotlib.pyplot as plt
# for i, td in enumerate(td_init):
#     fig, axs = plt.subplots(1,2, figsize=(11,5))
#     env.render(td, actions_untrained[i], ax=axs[0])
#     env.render(td, actions_trained[i], ax=axs[1])
#     axs[0].set_title(f"Untrained | Cost = {-rewards_untrained[i].item():.3f}")
#     axs[1].set_title(r"Trained $\pi_\theta$" + f"| Cost = {-out['reward'][i].item():.3f}")


