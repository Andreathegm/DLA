import torch
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from gymnasium.wrappers import RecordVideo
from policynet import PolicyNet
from reinforce import reinforce

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


seed = 1234
torch.manual_seed(seed)
np.random.seed(seed)

env = gym.make("CartPole-v1")
env.reset(seed=seed)

policy = PolicyNet(env).to(device)


rewards = reinforce(policy = policy,env = env,num_episodes=1000)
torch.save(policy.state_dict(), "weights/policy_final.pt")

plt.plot(rewards)
plt.savefig("plots/running_reward_training_curve.png")
plt.close()


env.close()

policy.eval()
env_video = gym.make("CartPole-v1", render_mode="rgb_array")
env_video = RecordVideo(env_video, video_folder="./videos", name_prefix="final")
obs, info = env_video.reset()
done = False

while not done:
    obs_tensor = torch.tensor(obs, dtype=torch.float32)
    action = policy(obs_tensor).argmax().item()
    obs, reward, terminated, truncated, info = env_video.step(action)
    done = terminated or truncated

env_video.close()
