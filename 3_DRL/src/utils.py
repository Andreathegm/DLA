import numpy as np
import torch
from torch.distributions import Categorical
import pygame
from src.device import device

_ = pygame.init()

## building a cache for gamma matrix for every episode lenght

_gamma_matrix_cache: dict[tuple, torch.Tensor] = {}

def compute_gamma_matrix(T , gamma ):
    key = (T, gamma)
    if key not in _gamma_matrix_cache:
        rows = torch.arange(T).unsqueeze(0)               # shape (1, T)
        cols = torch.arange(T).unsqueeze(1)               # shape (T, 1)
        mat  = torch.triu(float(gamma) ** (rows - cols))  # shape (T, T)
        _gamma_matrix_cache[key] = mat.to(device)

    return _gamma_matrix_cache[key]


def compute_returns(rewards , gamma ):

    r = torch.tensor(rewards, dtype=torch.float32).to(device)
    G = compute_gamma_matrix(len(rewards), gamma)   
    return G @ r                                    # shape (T,)


def select_action(env, obs,policy) -> tuple[int, torch.Tensor]:
    dist     = Categorical(policy(obs))
    action   = dist.sample()
    log_prob = dist.log_prob(action)
    ## with log prob we are basically saying the log_prob of that specific action we have taken
    return (action.item(), log_prob.reshape(1))


def run_episode(env, policy, maxlen: int = 500):
    observations = []
    actions      = []
    log_probs    = []
    rewards      = []

    (obs, info) = env.reset()
    for _ in range(maxlen):

        obs_tensor = torch.tensor(obs, dtype=torch.float32).to(device)

        (action, log_prob) = select_action(env, obs_tensor, policy)

        observations.append(obs_tensor)
        actions.append(action)
        log_probs.append(log_prob)

        (obs, reward, term, trunc, info) = env.step(action)
        rewards.append(reward)
        if term or trunc:
            break

    # concatenating  N scalar tensors (1,) into a unique tensor (T,)
    return (observations, actions, torch.cat(log_probs), rewards)


def evaluate_policy(env, policy, M: int) -> tuple[float, float]:
    policy.eval()
    total_rewards = []
    episode_lengths = []
    
    with torch.no_grad():
        for _ in range(M):
            (observations, actions, log_probs, rewards) = run_episode(env, policy)
            total_rewards.append(sum(rewards))
            episode_lengths.append(len(rewards))
    
    return (np.mean(total_rewards), np.mean(episode_lengths))