各文件说明：
154WTs.xls与HKY_WT.xlsx为两个风机坐标数据文件，前者154个风机，OS坐标155行；后者42风机，OS坐标在最后一行。

.ckpt后缀文件为训练出的模型，154风机的已在名字中标明，其余为42风机的模型，其中效果较好的是test_basic_100epochs.ckpt和test_withnocross.ckpt，分别对应不考虑交叉惩罚和考虑交叉惩罚。

test_settings.py用于测试cuda，torch，以及pandas是否正确安装

test_rl4co.py测试能否正常用rl4co库求解cvrp问题

test_mtvrp.py是调试时的文件，目前无用

test_rl4ecsp.py是用于训练自己模型的文件。
主要进行了修改的文件还有：
CVRP的env，可通过test_rl4ecsp.py中env = CVRPEnv(generator_params=generator_params)，按住ctrl点CVRPEnv打开。
CVRP的generator，可通过generator = CVRPGenerator(**generator_params)打开。
