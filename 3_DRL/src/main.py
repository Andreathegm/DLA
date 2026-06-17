import torch
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from gymnasium.wrappers import RecordVideo
from policynet import PolicyNet
from reinforce import reinforce
from device import device
from evaluate import record_video_evaluation

seed = 1234
torch.manual_seed(seed)
np.random.seed(seed)

env = gym.make("CartPole-v1")
env.reset(seed=seed)

policy = PolicyNet(env).to(device)

running_rewards,eval_rewards,eval_lenghts = reinforce(policy=policy, env=env, num_episodes=1000)
torch.save(policy.state_dict(), "weights/policy_final.pt")

plt.plot(running_rewards)
plt.savefig("plots/running_reward_training_curve.png")
plt.close()

env.close()

# ── Valutazione finale con video ─────────────────────────────────────────────
policy.eval()
record_video_evaluation(policy = policy)