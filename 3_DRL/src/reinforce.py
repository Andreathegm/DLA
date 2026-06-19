import torch
import pygame
from src.utils import run_episode, compute_returns , evaluate_policy

_ = pygame.init()


def reinforce(policy, env, baseline , lr=1e-2 ,env_render=None, gamma=0.99, num_episodes=10, optimizer=None , N = 50 , M = 20 , w_eval_path = "weights/eval",loss_op = "mean"):

    opt = optimizer if optimizer else torch.optim.Adam(policy.parameters(), lr=lr)

    ## Metrics
    running_rewards = [0.0]
    eval_avg_rewards = []
    eval_avg_lengths = []
    eval_best_reward = 0.0

    policy.train()
    for episode in range(num_episodes):
        (observations, actions, log_probs, rewards) = run_episode(env, policy)

        returns = compute_returns(rewards, gamma)

        running_rewards.append(0.05 * returns[0].item() + 0.95 * running_rewards[-1])

        # standardization
        #returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        returns = returns_with_baseline(returns,baseline)

        ## its equal to write
        ## 1/sigma*    (returns - returns.mean()) isolating the difference , also
        ## the mean of the return is indipendent of the state

        opt.zero_grad()

        if loss_op == "mean":
            loss = -(log_probs * returns).mean()
        elif loss_op == "sum":
            loss = -(log_probs * returns).sum()
        
        loss.backward()
        opt.step()

        if not episode % 100:
            if env_render:
                policy.eval()
                run_episode(env_render, policy)
                policy.train()
            print(f'Running reward: {running_rewards[-1]:.4f}')


        ## policy evaluation every N steps and checkpointing
        if episode > 0 and episode % N == 0:
            avg_reward, avg_length = evaluate_policy(env=env, policy=policy, M=M)
            if avg_reward > eval_best_reward:

                torch.save(policy.state_dict(),f"{w_eval_path}/best_policy.pt")
                eval_best_reward = avg_reward
            
            eval_avg_rewards.append(avg_reward)
            eval_avg_lengths.append(avg_length)
            policy.train()

    policy.eval()
    return running_rewards,eval_avg_rewards,eval_avg_lengths

def returns_with_baseline(returns, baseline=None, epsilon=1e-8):

    if baseline is None:
        return returns

    if baseline == "mean":
        b = returns.mean()
        std = returns.std() + epsilon
        return (returns - b) / std

    if callable(baseline):
        pass

    # fallback: valore numerico fisso
    return returns - baseline