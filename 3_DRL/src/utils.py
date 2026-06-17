import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from device import device

# Plus one non standard one -- we need this to sample from policies.
from torch.distributions import Categorical

# This *might* help PyGame crash less...
import pygame
_ = pygame.init()

# Given an environment, observation, and policy, sample from pi(a | obs). Returns the
# selected action and the log probability of that action (needed for policy gradient).
def select_action(env, obs, policy):
    dist = Categorical(policy(obs))
    action = dist.sample()
    log_prob = dist.log_prob(action).to(device)
    return (action.item(), log_prob.reshape(1))

# Utility to compute the discounted total reward. Torch doesn't like flipped arrays, so we need to
# .copy() the final numpy array. There's probably a better way to do this.
def compute_gamma_matrix(T,gamma):
    rows = torch.arange(T).unsqueeze(0)
    columns = torch.arange(T).unsqueeze(1)
    return torch.triu(float(gamma) ** (rows- columns))

def compute_returns(rewards, gamma):
    rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
    gamma_matrix = compute_gamma_matrix(len(rewards),gamma).to(device)
    return gamma_matrix @ rewards
    #return np.flip(np.cumsum([gamma**(i+1)*r for (i, r) in enumerate(rewards)][::-1]), 0).copy()

# Given an environment and a policy, run it up to the maximum number of steps.
def run_episode(env, policy, maxlen=500):
    observations = []
    actions = []
    log_probs = []
    rewards = []
    
    # Reset the environment and start the episode.
    (obs, info) = env.reset()
    for i in range(maxlen):
        # Get the current observation, run the policy and select an action.
        obs = torch.tensor(obs,dtype= torch.float32).to(device)
        (action, log_prob) = select_action(env, obs, policy)
        observations.append(obs)
        actions.append(action)
        log_probs.append(log_prob)
        
        # Advance the episode by executing the selected action.
        (obs, reward, term, trunc, info) = env.step(action)
        rewards.append(reward)
        if term or trunc:
            break
    return (observations, actions, torch.cat(log_probs), rewards)