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
    device = rtsUtils.getTorchDevice(args) #get device

    envs = envInitializer.envInitializer(args)
    
    agent = agentSetup.AgentSmall(envs).to(device) # chose between agentSmall and AgentLarge, 200k vs 800k paramenters in NN architecture

    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    writer = SummaryWriter(f"{args.experiment_name}/{args.experiment_name}_log")
    writer.add_text("hyperparameters", "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])))

    model = maskedPPO.PPOTrainer(args,envs,agent,writer,optimizer,device)
    model.train()

    envs.close()
    writer.close()
