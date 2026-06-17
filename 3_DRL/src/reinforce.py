import numpy as np
import matplotlib.pyplot as plt
import gymnasium
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import pygame
from device import device

from utils import run_episode,compute_returns

_ = pygame.init()

def reinforce(policy, env, env_render=None, gamma=0.99, num_episodes=10 , optimizer = None):
    if not optimizer :

        # The only non-vanilla part: we use Adam instead of SGD.
        opt = torch.optim.Adam(policy.parameters(), lr=1e-2)
    else :
        opt = optimizer

    # Track episode rewards in a list.
    running_rewards = [0.0]
    
    # The main training loop.
    policy.train()
    for episode in range(num_episodes):
        # Run an episode of the environment, collect everything needed for policy update.
        (observations, actions, log_probs, rewards) = run_episode(env, policy)
        
        # Compute the discounted reward for every step of the episode. 
        returns = compute_returns(rewards, gamma)

        returns = returns.to(device)
        log_probs = log_probs.to(device)
        
        # Keep a running average of total discounted rewards for the whole episode.
        running_rewards.append(0.05 * returns[0].item() + 0.95 * running_rewards[-1])
        
        # Standardize returns.
        returns = (returns - returns.mean()) / returns.std()
        
        # Make an optimization step
        opt.zero_grad()
        loss = (-log_probs * returns).mean()
        loss.backward()
        opt.step()
        
        # Render an episode after every 100 policy updates.
        if not episode % 100:
            if env_render:
                policy.eval()
                run_episode(env_render, policy)
                policy.train()
            print(f'Running reward: {running_rewards[-1]}')
    
    # Return the running rewards.
    policy.eval()
    return running_rewards


