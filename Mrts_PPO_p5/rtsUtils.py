import os
import torch

def saveModel(model, args, update, globalStep, experiment_name):
  if (update - 1) % args.save_frequency == 0:
      if not os.path.exists(f"models/{experiment_name}"):
          os.makedirs(f"models/{experiment_name}")
      torch.save(model.state_dict(), f"models/{experiment_name}/agent.pt")
      torch.save(model.state_dict(), f"models/{experiment_name}/{globalStep}.pt")

def printModelParams(agent):
    print("Model's state_dict:")
    for param_tensor in agent.state_dict():
        print(param_tensor, "\t", agent.state_dict()[param_tensor].size())
    total_params = sum([param.nelement() for param in agent.parameters()])
    print("Model's total parameters:", total_params)


def allocTensorToDevice(args,envs,device,action_space_shape,invalid_action_shape):
    obs = torch.empty((args.num_steps, args.num_envs) + envs.observation_space.shape).to(device)
    actions = torch.empty((args.num_steps, args.num_envs) + action_space_shape).to(device)
    logprobs = torch.empty((args.num_steps, args.num_envs)).to(device)
    rewards = torch.empty((args.num_steps, args.num_envs)).to(device)
    dones = torch.empty((args.num_steps, args.num_envs)).to(device)
    values = torch.empty((args.num_steps, args.num_envs)).to(device)
    invalid_action_masks = torch.empty((args.num_steps, args.num_envs) + invalid_action_shape).to(device)

    return obs, actions, logprobs, rewards, dones, values, invalid_action_masks

def calculateActionShapes(mapsize, envs):
    action_space_shape = (mapsize, len(envs.action_plane_space.nvec))
    invalid_action_shape = (mapsize, envs.action_plane_space.nvec.sum())
    return action_space_shape, invalid_action_shape

def getTorchDevice(args):
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    print(f"Device: {device} for this session")
    return device

def calculateWinRate(rounds,modelScore,aiScore, result):
    if(result == "-1"):
        aiScore += 1
    else:
        modelScore +=1
    
    #there are draws as well hence 1-winrate is not used to calc the other's winrate
    winrateModel = modelScore/rounds
    winrateAI = aiScore/rounds

    print("AI winrate: ", winrateAI)
    print("Model winrate: ", winrateModel)
    return modelScore, aiScore
    




