# http://proceedings.mlr.press/v97/han19a/han19a.pdf

import argparse
import os
import random
import time
from distutils.util import strtobool

from distutils.util import strtobool
import numpy as np
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical


import numpy as np
import torch
from gym.spaces import MultiDiscrete
from stable_baselines3.common.vec_env import VecMonitor, VecVideoRecorder
from torch.utils.tensorboard import SummaryWriter

from gym_microrts import microrts_ai  # noqa


import sys
sys.path.append('../')  # Add the outer folder to the sys.path 
import agentSetup
import evalArgParser
import envInitializer




if __name__ == "__main__":
    args = evalArgParser.parse_args()

    # TRY NOT TO MODIFY: setup the environment
    experiment_name = f"{args.gym_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    writer = SummaryWriter(f"runs/{experiment_name}")
    writer.add_text( "hyperparameters", "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])))

    # TRY NOT TO MODIFY: seeding
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    envs = envInitializer.envInitializer(args,ai2s=[eval(f"microrts_ai.{args.ai}")])
    model = agentSetup.Agent(envs).to(device)
    invalid_action_shape = (args.mapsize, envs.action_plane_space.nvec.sum())

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    # Note how `next_obs` and `next_done` are used; their usage is equivalent to
    # https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail/blob/84a7582477fb0d5c82ad6d850fe476829dddd2e1/a2c_ppo_acktr/storage.py#L60
    next_obs = torch.Tensor(envs.reset()).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    ## CRASH AND RESUME LOGIC:
    starting_update = 1
    model.load_state_dict(torch.load(args.agent_model_path, map_location=device))
    model.eval()
    


    for update in range(starting_update, args.num_updates + 1):
        # TRY NOT TO MODIFY: prepare the execution of the game.
        for step in range(0, args.num_steps):
            envs.render()
            global_step += 1 * args.num_envs
            # ALGO LOGIC: put action logic here
            with torch.no_grad():
                invalid_action_masks = torch.tensor(np.array(envs.get_action_mask())).to(device)

                if args.ai:
                    action, logproba, _, _, vs = model.get_action_and_value(
                        next_obs, envs=envs, invalid_action_masks=invalid_action_masks, device=device
                    )
            try:
                next_obs, rs, ds, infos = envs.step(action.cpu().numpy().reshape(envs.num_envs, -1))
                next_obs = torch.Tensor(next_obs).to(device)
            except Exception as e:
                e.printStackTrace()
                raise

            for idx, info in enumerate(infos):
                if "episode" in info.keys():
                    if args.ai:
                        print("against", args.ai, info["microrts_stats"]["WinLossRewardFunction"])
                   

    envs.close()
    writer.close()
