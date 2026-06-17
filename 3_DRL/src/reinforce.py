import torch
from torch.distributions import Categorical
import pygame
from device import device
from utils import run_episode, compute_returns , evaluate_policy

_ = pygame.init()


def reinforce(policy, env, env_render=None, gamma=0.99, num_episodes=10, optimizer=None , N = 50 , M = 20 ):
    opt = optimizer if optimizer else torch.optim.Adam(policy.parameters(), lr=1e-2)

    ## Metrics
    running_rewards = [0.0]
    eval_avg_rewards = []
    eval_avg_lengths = []
    eval_best_reward = 0.0

    policy.train()
    for episode in range(num_episodes):
        (observations, actions, log_probs, rewards) = run_episode(env, policy)

        # ── Perché NON servono .to(device) qui ──────────────────────────────
        # returns: compute_returns() fa già il .to(device) internamente
        #          (sul tensore rewards) e usa la gamma matrix già su device.
        #          Il risultato è già su device.
        #
        # log_probs: torch.cat(log_probs) in run_episode concatena tensori che
        #            sono già su device (select_action non fa più .to()).
        #            Anche log_probs è già su device.
        #
        # Quindi i due .to(device) originali in reinforce() erano ridondanti:
        # spostavano su device dati che erano già lì, pagando inutilmente
        # l'overhead del controllo (anche se no-op, viene chiamato 1000 volte).
        # ────────────────────────────────────────────────────────────────────
        returns = compute_returns(rewards, gamma)

        running_rewards.append(0.05 * returns[0].item() + 0.95 * running_rewards[-1])

        # Standardizza i returns (tutto già su device, operazione locale).
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        # Nota: aggiunto +1e-8 per evitare divisione per zero negli episodi
        # con reward costante (es. primo episodio molto corto).

        opt.zero_grad()
        loss = (-log_probs * returns).mean()
        loss.backward()
        opt.step()

        if not episode % 100:
            if env_render:
                policy.eval()
                run_episode(env_render, policy)
                policy.train()
            print(f'Running reward: {running_rewards[-1]:.4f}')

        if episode > 0 and episode % N == 0:
            avg_reward, avg_length = evaluate_policy(env=env, policy=policy, M=M)
            if avg_reward > eval_best_reward:
                torch.save(policy.state_dict(), "weights/eval/best_policy.pt")
                eval_best_reward = avg_reward
            eval_avg_rewards.append(avg_reward)
            eval_avg_lengths.append(avg_length)
            policy.train()




    policy.eval()
    return running_rewards,eval_avg_rewards,eval_avg_lengths