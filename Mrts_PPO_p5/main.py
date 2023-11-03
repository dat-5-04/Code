
import os
import random

import time
from distutils.util import strtobool
from typing import List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter

import argParserInit
import agentSetup
import envInitializer
import rtsUtils
import maskedPPO



if __name__ == "__main__":
    #Parsed arguments
    args = argParserInit.parse_args()

    #Everything optional for the train
    experiment_name = f"{args.gym_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    device = rtsUtils.getTorchDevice(args) #get device
    envs = envInitializer.envInitializer(args=args)
    agent = agentSetup.Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    writer = SummaryWriter(f"log/{experiment_name}")
    writer.add_text("hyperparameters", "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])))

    model = maskedPPO.PPOTrainer(args,envs,agent,writer,optimizer,device,"modelNameHere")
    model.train()
    
    envs.close()
    writer.close()
