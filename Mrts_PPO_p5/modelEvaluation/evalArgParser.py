import argparse
import os

import time
from distutils.util import strtobool


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp-name', type=str, default=os.path.basename(__file__).rstrip(".py"),
        help='the name of this experiment')
    parser.add_argument('--gym-id', type=str, default="MicroRTSGridModeVecEnv",
        help='the id of the gym environment')
    parser.add_argument('--learning-rate', type=float, default=2.5e-4,
        help='the learning rate of the optimizer')
    parser.add_argument('--seed', type=int, default=1,
        help='seed of the experiment')
    parser.add_argument('--total-timesteps', type=int, default=1000000,
        help='total timesteps of the experiments')
    parser.add_argument('--torch-deterministic', type=lambda x: bool(strtobool(x)), default=True, nargs='?', const=True,
        help='if toggled, `torch.backends.cudnn.deterministic=False`')
    parser.add_argument('--cuda', type=lambda x: bool(strtobool(x)), default=True, nargs='?', const=True,
        help='if toggled, cuda will not be enabled by default')
    parser.add_argument('--prod-mode', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True,
        help='run the script in production mode and use wandb to log outputs')
    parser.add_argument('--capture-video', type=lambda x: bool(strtobool(x)), default=True, nargs='?', const=True,
        help='whether to capture videos of the agent performances (check out `videos` folder)')
    parser.add_argument('--wandb-project-name', type=str, default="cleanRL",
        help="the wandb's project name")
    parser.add_argument('--wandb-entity', type=str, default=None,
        help="the entity (team) of wandb's project")

    # Algorithm specific arguments
    parser.add_argument('--partial-obs', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True,
        help='if toggled, the game will have partial observability')
    parser.add_argument('--num-steps', type=int, default=256,
        help='the number of steps per game environment')
    parser.add_argument("--agent-model-path", type=str, default="models/modelNameHere/agent.pt",
        help="the path to the agent's model")
    parser.add_argument('--ai', type=str, default="coacAI",
        help='the opponent AI to evaluate against')
    parser.add_argument('--model-type', type=str, default=f"ppo_gridnet", choices=["ppo_gridnet_large", "ppo_gridnet"],
        help='the output path of the leaderboard csv')
    parser.add_argument('--train-maps', nargs='+', default=["maps/8x8/basesWorkers8x8.xml"],
        help='the list of maps used during training')
    parser.add_argument('--gamma', type=float, default=0.99,
        help='the discount factor gamma')
    args = parser.parse_args()
    if not args.seed:
        args.seed = int(time.time())
    if args.ai:
        args.num_bot_envs, args.num_selfplay_envs = 1, 0
    else:
        args.num_bot_envs, args.num_selfplay_envs = 0, 2
    args.num_envs = args.num_selfplay_envs + args.num_bot_envs
    args.batch_size = int(args.num_envs * args.num_steps)
    args.num_updates = args.total_timesteps // args.batch_size
    args.mapsize = 8*8
    # fmt: on
    return args
