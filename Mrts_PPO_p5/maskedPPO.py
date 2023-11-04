import torch
import numpy as np
import time
import random
import torch.nn as nn

import rtsUtils

class PPOTrainer:
    def __init__(self, args, envs, agent, writer, optimizer, device, printParams = True):
        self.args = args
        self.envs = envs
        self.agent = agent
        self.writer = writer
        self.optimizer = optimizer
        self.device = device
        

        #init of learnign rate lambda, device spaces etc. 
        self.lr = lambda f: f * self.args.learning_rate
        self.action_space_shape = None
        self.invalid_action_shape = None
        self.obs = None
        self.actions = None
        self.logprobs = None
        self.rewards = None
        self.dones = None
        self.values = None
        self.invalid_action_masks = None

        #global params for this ppo case
        self.global_step = None
        self.next_obs = None
        self.next_done = None
        self.start_time = None
        self.starting_update = None

        if printParams:
            rtsUtils.printModelParams(self.agent)

    def annealLearningRate(self,update):
         if  self.args.anneal_lr:
          frac = 1.0 - (update - 1.0) /  self.args.num_updates
          lrnow =  self.lr(frac)
          self.optimizer.param_groups[0]["lr"] = lrnow
    
    def collectRollOuts(self):
        for step in range(0,  self.args.num_steps):
            # envs.render()
            self.global_step +=  self.args.num_envs #original 
            self.obs[step] = self.next_obs
            self.dones[step] = self.next_done
            # ALGO LOGIC: put action logic here
            with torch.no_grad():
                self.invalid_action_masks[step] = torch.tensor(self.envs.get_action_mask()).to(self.device)
                action, logproba, _, _, vs = self.agent.get_action_and_value(self.next_obs, envs=self.envs, invalid_action_masks=self.invalid_action_masks[step], device=self.device)
                self.values[step] = vs.flatten()

            self.actions[step] = action
            self.logprobs[step] = logproba
            try:
                self.next_obs, rs, ds, infos = self.envs.step(action.cpu().numpy().reshape(self.envs.num_envs, -1))
                self.next_obs = torch.Tensor(self.next_obs).to(self.device)
            except Exception as e:
                e.printStackTrace()
                raise
            self.rewards[step], self.next_done = torch.Tensor(rs).to(self.device), torch.Tensor(ds).to(self.device)

            for info in infos:
                if "episode" in info.keys():
                    print(f"global_step={self.global_step}, episodic_return={info['episode']['r']}")
                    self.writer.add_scalar("charts/episodic_return", info["episode"]["r"], self.global_step)
                    self.writer.add_scalar("charts/episodic_length", info["episode"]["l"], self.global_step)
                    for key in info["microrts_stats"]:
                        self.writer.add_scalar(f"charts/episodic_return/{key}", info["microrts_stats"][key], self.global_step)
                    break

    def calculateAdvantage(self):
        with torch.no_grad():
            last_value = self.agent.get_value(self.next_obs).reshape(1, -1)
            if self.args.gae:
                advantages = torch.zeros_like(self.rewards).to(self.device)
                lastgaelam = 0
                for t in reversed(range(self.args.num_steps)):
                    if t == self.args.num_steps - 1:
                        nextnonterminal = 1.0 - self.next_done
                        nextvalues = last_value
                    else:
                        nextnonterminal = 1.0 - self.dones[t + 1]
                        nextvalues = self.values[t + 1]
                    delta = self.rewards[t] + self.args.gamma * nextvalues * nextnonterminal - self.values[t]
                    advantages[t] = lastgaelam = delta + self.args.gamma * self.args.gae_lambda * nextnonterminal * lastgaelam
                returns = advantages + self.values
            else:
                returns = torch.zeros_like(self.rewards).to(self.device)
                for t in reversed(range(self.args.num_steps)):
                    if t == self.args.num_steps - 1:
                        nextnonterminal = 1.0 - self.next_done
                        next_return = last_value
                    else:
                        nextnonterminal = 1.0 -  self.dones[t + 1]
                        next_return = returns[t + 1]
                    returns[t] = self.rewards[t] + self.args.gamma * nextnonterminal * next_return
                advantages = returns - self.values
        return advantages, returns
    
    def flatten_tensors(self, advantages, returns):
        b_obs = self.obs.reshape((-1,) + self.envs.observation_space.shape)
        b_logprobs = self.logprobs.reshape(-1)
        b_actions = self.actions.reshape((-1,) + self.action_space_shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = self.values.reshape(-1)
        b_invalid_action_masks = self.invalid_action_masks.reshape((-1,) + self.invalid_action_shape)
        
        return b_obs, b_logprobs, b_actions, b_advantages, b_returns, b_values, b_invalid_action_masks

    

    def optimizePolicyAndVal(self,b_advantages, b_obs,b_actions, b_invalid_action_masks,b_logprobs, b_values, b_returns):
        inds = np.arange(self.args.batch_size,)
        for i_epoch_pi in range(self.args.update_epochs):
            np.random.shuffle(inds)
            for start in range(0, self.args.batch_size, self.args.minibatch_size):
                end = start + self.args.minibatch_size
                minibatch_ind = inds[start:end]
                mb_advantages = b_advantages[minibatch_ind]
                if self.args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
                _, newlogproba, entropy, _, new_values = self.agent.get_action_and_value(
                    b_obs[minibatch_ind], b_actions.long()[minibatch_ind], b_invalid_action_masks[minibatch_ind], self.envs,  self.device
                )
                ratio = (newlogproba - b_logprobs[minibatch_ind]).exp()

                # Stats
                approx_kl = (b_logprobs[minibatch_ind] - newlogproba).mean()

                # Policy loss - PPO clipping
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.args.clip_coef, 1 + self.args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                entropy_loss = entropy.mean()

                # Value loss
                new_values = new_values.view(-1)
                if self.args.clip_vloss:
                    v_loss_unclipped = (new_values - b_returns[minibatch_ind]) ** 2
                    v_clipped = b_values[minibatch_ind] + torch.clamp( new_values - b_values[minibatch_ind], -self.args.clip_coef, self.args.clip_coef)
                    v_loss_clipped = (v_clipped - b_returns[minibatch_ind]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((new_values - b_returns[minibatch_ind]) ** 2)

                loss = pg_loss - self.args.ent_coef * entropy_loss + v_loss * self.args.vf_coef

                self.optimizer.zero_grad(set_to_none=True) #less mem operations to "zero-grad by setting to none instead of zero"
                loss.backward()
                nn.utils.clip_grad_norm_(self.agent.parameters(), self.args.max_grad_norm)
                self.optimizer.step()

        return v_loss, pg_loss, entropy, approx_kl, i_epoch_pi
          
    def writeLog(self,update,v_loss,pg_loss,entropy,approx_kl,i_epoch_pi):
         # TRY NOT TO MODIFY: record rewards for plotting purposes
        self.writer.add_scalar("charts/learning_rate",  self.optimizer.param_groups[0]["lr"], self.global_step)
        self.writer.add_scalar("charts/update", update, self.global_step)
        self.writer.add_scalar("losses/value_loss", v_loss.detach().item(), self.global_step)
        self.writer.add_scalar("losses/policy_loss", pg_loss.detach().item(), self.global_step)
        self.writer.add_scalar("losses/entropy", entropy.detach().mean().item(), self.global_step)
        self.writer.add_scalar("losses/approx_kl", approx_kl.detach().item(), self.global_step)
        if self.args.kle_stop or self.args.kle_rollback:
            self.writer.add_scalar("debug/pg_stop_iter", i_epoch_pi, self.global_step)
        self.writer.add_scalar("charts/sps", int( self.global_step / (time.time() -  self.start_time)), self.global_step)
        print("SPS:", int(self.global_step / (time.time() -  self.start_time)))

    def train(self):
        #init Seeds
        random.seed(self.args.seed)
        np.random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)
        torch.backends.cudnn.deterministic = self.args.torch_deterministic #check dif between true and false here. 

        #init of params, issue when initialized in constructor
        self.action_space_shape, self.invalid_action_shape = rtsUtils.calculateActionShapes(self.args.mapsize, self.envs) 
        self.obs, self.actions, self.logprobs, self.rewards, self.dones, self.values, self.invalid_action_masks = rtsUtils.allocTensorToDevice(self.args, self.envs, self.device, self.action_space_shape, self.invalid_action_shape)
        self.global_step = 0
        self.start_time = time.time()
        self.next_obs = torch.Tensor(self.envs.reset()).to(self.device)
        self.next_done = torch.zeros(self.args.num_envs).to(self.device)
        self.starting_update = 1

      
        for update in range( self.starting_update,  self.args.num_updates + 1):
            # Annealing the rate if instructed to do so(reduce reward with respect to current steps)
            self.annealLearningRate(update)

            #collect rollouts for for gradient calc  etc. 
            self.collectRollOuts() 

            # bootstrap reward if not done. reached the batch limit
            advantages,returns = self.calculateAdvantage()

            #flatten batch tensors
            b_obs, b_logprobs, b_actions, b_advantages, b_returns, b_values, b_invalid_action_masks = self.flatten_tensors(advantages,returns)
            
            # Optimizing the policy and value network
            v_loss,pg_loss,entropy,approx_kl,i_epoch_pi = self.optimizePolicyAndVal(b_advantages, b_obs, b_actions, b_invalid_action_masks,b_logprobs, b_values, b_returns)
            
            #save model with frequency as arg
            rtsUtils.saveModel(self.agent,self.args,update,self.global_step,"testName")

            #write the current update to the log so it is accesible from TF
            self.writeLog(update,v_loss,pg_loss,entropy,approx_kl,i_epoch_pi)
