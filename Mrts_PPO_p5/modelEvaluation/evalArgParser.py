import argparse
import numpy as np
from distutils.util import strtobool
from gym_microrts import microrts_ai


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--total-timesteps', type=int, default=1000000,
        help='total timesteps of the experiments')
    parser.add_argument('--torch-deterministic', type=lambda x: bool(strtobool(x)), default=True, nargs='?', const=True,
        help='if toggled, `torch.backends.cudnn.deterministic=False`')
    parser.add_argument('--cuda', type=lambda x: bool(strtobool(x)), default=True, nargs='?', const=True,
        help='if toggled, cuda will not be enabled by default')

    # Algorithm specific arguments
    parser.add_argument('--partial-obs', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True,
        help='if toggled, the game will have partial observability')
    parser.add_argument("--agent-model-path", type=str, default=f"models/agent.pt",
        help="the path to the agent's model")
    parser.add_argument('--ai', type=str, default="coacAI",
        help='the opponent AI to evaluate against')
    parser.add_argument('--train-maps', nargs='+', default=["maps/16x16/basesWorkers16x16A.xml"],
        help='the list of maps used during training')
    parser.add_argument('--gamma', type=float, default=0.99,
        help='the discount factor gamma')
    args = parser.parse_args()

    args.selfplay = False
    if  args.selfplay:
        args.num_bot_envs, args.num_selfplay_envs = 1, 0    
    else:
        args.num_bot_envs, args.num_selfplay_envs = 0, 2

    args.seed = 324343
    args.num_steps = 512
    args.num_envs = args.num_selfplay_envs + args.num_bot_envs
    args.batch_size = int(args.num_envs * args.num_steps)
    args.num_updates = args.total_timesteps // args.batch_size
    args.mapsize = 16*16
    args.experiment_name = "eval"
    args.record_video = False
    args.ai2s= [ (eval(f"microrts_ai.{args.ai}")) if args.selfplay else 0 ]
    args.reward_weights = np.array([10.0, 1.0, 1.0, 0.2, 1.0, 4.0])
    args.max_env_steps =2048
    #std_paper_plus_Reduced_worker_reward vs worker
    return args
