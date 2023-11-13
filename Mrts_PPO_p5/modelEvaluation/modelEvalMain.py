
import random
import time

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import agentSetup
import evalArgParser
import envInitializer
import rtsUtils
import sys
sys.path.append('../')  # Add the outer folder to the sys.path 


if __name__ == "__main__":
    args = evalArgParser.parse_args()

    # TRY NOT TO MODIFY: setup the environment
    writer = SummaryWriter(f"runs/{args.experiment_name}")
    writer.add_text( "hyperparameters", "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])))

    # TRY NOT TO MODIFY: seeding
    device = rtsUtils.getTorchDevice(args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    envs = envInitializer.envInitializer(args)
    model = agentSetup.Agent(envs).to(device)
    invalid_action_shape = (args.mapsize, envs.action_plane_space.nvec.sum())

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs = torch.Tensor(envs.reset()).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    ## CRASH AND RESUME LOGIC:
    starting_update = 1
    model.load_state_dict(torch.load(args.agent_model_path, map_location=device))
    model.eval()
    rounds = 0
    modelScore = 0
    aiScore = 0

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
                       rounds += 1
                       modelScore, aiScore = rtsUtils.calculateWinRate(rounds, modelScore, aiScore, info["microrts_stats"]["WinLossRewardFunction"])


    envs.close()
    writer.close()
