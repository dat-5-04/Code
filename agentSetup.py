
from distutils.util import strtobool
import numpy as np
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical

# ALGO LOGIC: initialize agent here:
class CategoricalMasked(Categorical):
    def __init__(self, probs=None, logits=None, validate_args=None, masks=[], mask_value=None):
        logits = torch.where(masks.bool(), logits, mask_value)
        super(CategoricalMasked, self).__init__(probs, logits, validate_args)


class Transpose(nn.Module):
    def __init__(self, permutation):
        super().__init__()
        self.permutation = permutation

    def forward(self, x):
        return x.permute(self.permutation)


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs, mapsize=8 * 8):
        super(Agent, self).__init__()
        self.mapsize = mapsize
        h, w, c = envs.observation_space.shape
        self.encoder = nn.Sequential(
            Transpose((0, 3, 1, 2)),
            layer_init(nn.Conv2d(c, 32, kernel_size=3, padding=1)),
            nn.MaxPool2d(3, stride=2, padding=1),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, kernel_size=3, padding=1)),
            nn.MaxPool2d(3, stride=2, padding=1),
            nn.ReLU(),
        )

        # Calculate the output size of the encoder dynamically based on the input size.
        encoder_output_size = self.calculate_encoder_output_size(h, w)

        self.actor = nn.Sequential(
            layer_init(nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1)),
            nn.ReLU(),
            layer_init(nn.ConvTranspose2d(32, 78, 3, stride=2, padding=1, output_padding=1)),
            Transpose((0, 2, 3, 1)),
        )

        # Adjust the critic network based on the dynamic encoder output size.
        self.critic = nn.Sequential(
            nn.Flatten(),
            layer_init(nn.Linear(encoder_output_size, 128)),
            nn.ReLU(),
            layer_init(nn.Linear(128, 1), std=1),
        )

        self.register_buffer("mask_value", torch.tensor(-1e8))

    def calculate_encoder_output_size(self, h, w):
        # Calculate the output size of the encoder dynamically.
        for layer in self.encoder:
            if isinstance(layer, nn.Conv2d):
                h = ((h + 2 * layer.padding[0] - layer.dilation[0] * (layer.kernel_size[0] - 1) - 1) // layer.stride[0]) + 1
                w = ((w + 2 * layer.padding[1] - layer.dilation[1] * (layer.kernel_size[1] - 1) - 1) // layer.stride[1]) + 1
            elif isinstance(layer, nn.MaxPool2d):
                h = (h + 2 * layer.padding - layer.dilation * (layer.kernel_size - 1) - 1) // layer.stride + 1
                w = (w + 2 * layer.padding - layer.dilation * (layer.kernel_size - 1) - 1) // layer.stride + 1

        return 64 * h * w

    def get_action_and_value(self, x, action=None, invalid_action_masks=None, envs=None, device=None):
        hidden = self.encoder(x)
        logits = self.actor(hidden)
        grid_logits = logits.reshape(-1, envs.action_plane_space.nvec.sum())
        split_logits = torch.split(grid_logits, envs.action_plane_space.nvec.tolist(), dim=1)

        if action is None:
            invalid_action_masks = invalid_action_masks.view(-1, invalid_action_masks.shape[-1])
            split_invalid_action_masks = torch.split(invalid_action_masks, envs.action_plane_space.nvec.tolist(), dim=1)
            multi_categoricals = [
                CategoricalMasked(logits=logits, masks=iam, mask_value=self.mask_value)
                for (logits, iam) in zip(split_logits, split_invalid_action_masks)
            ]
            action = torch.stack([categorical.sample() for categorical in multi_categoricals])
        else:
            invalid_action_masks = invalid_action_masks.view(-1, invalid_action_masks.shape[-1])
            action = action.view(-1, action.shape[-1]).T
            split_invalid_action_masks = torch.split(invalid_action_masks, envs.action_plane_space.nvec.tolist(), dim=1)
            multi_categoricals = [
                CategoricalMasked(logits=logits, masks=iam, mask_value=self.mask_value)
                for (logits, iam) in zip(split_logits, split_invalid_action_masks)
            ]


        logprob = torch.stack([categorical.log_prob(a) for a, categorical in zip(action, multi_categoricals)])
        entropy = torch.stack([categorical.entropy() for categorical in multi_categoricals])
        num_predicted_parameters = len(envs.action_plane_space.nvec)
        logprob = logprob.T.view(-1, self.mapsize, num_predicted_parameters)
        entropy = entropy.T.view(-1, self.mapsize, num_predicted_parameters)
        action = action.T.view(-1, self.mapsize, num_predicted_parameters)
        return action, logprob.sum(1).sum(1), entropy.sum(1).sum(1), invalid_action_masks, self.critic(hidden)

    def get_value(self, x):
        return self.critic(self.encoder(x))