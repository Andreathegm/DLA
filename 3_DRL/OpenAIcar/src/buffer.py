import torch
import numpy as np

class RolloutBuffer:
    def __init__(self, num_steps, num_envs, obs_shape, action_shape, device):
        self.num_steps = num_steps
        self.num_envs = num_envs
        self.device = device
        
        batch_shape = (num_steps,num_envs)
        
        self.obs = torch.zeros(batch_shape + obs_shape).to(device)        ## (T,num_envs,(obs_shape)) Time-Major format
        self.actions = torch.zeros(batch_shape + action_shape).to(device)
        self.logprobs = torch.zeros(batch_shape).to(device)
        self.rewards = torch.zeros(batch_shape).to(device)
        self.dones = torch.zeros(batch_shape).to(device)
        self.values = torch.zeros(batch_shape).to(device)
        
        self.step = 0

    def add(self, obs, action, logprob, reward, done, value):
        self.obs[self.step] = obs
        self.actions[self.step] = action
        self.logprobs[self.step] = logprob
        self.rewards[self.step] = reward
        self.dones[self.step] = done
        self.values[self.step] = value
        self.step = (self.step + 1) % self.num_steps  ## TODO: how con we handle this

    def compute_returns_and_advantages(self, next_value, next_done, gamma, gae_lambda):

        ## next done is (N,)
        advantages = torch.zeros_like(self.rewards).to(self.device)
        lastgaelam = 0
        for t in reversed(range(self.num_steps)):
            if t == self.num_steps - 1:
                nextnonterminal = 1.0 - next_done
                nextvalues = next_value
            else:
                nextnonterminal = 1.0 - self.dones[t + 1]
                nextvalues = self.values[t + 1]
            delta = self.rewards[t] + gamma * nextvalues * nextnonterminal - self.values[t]        ## temporal differnce
            advantages[t] = lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
        
        returns = advantages + self.values
        return returns, advantages

    def get_flattened_data(self, returns, advantages):
        ## basically unites n_episodes and num_envs
        b_obs = self.obs.reshape((-1,) + self.obs.shape[2:])
        b_logprobs = self.logprobs.reshape(-1)
        b_actions = self.actions.reshape((-1,) + self.actions.shape[2:])
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = self.values.reshape(-1)
        
        return b_obs, b_logprobs, b_actions, b_advantages, b_returns, b_values