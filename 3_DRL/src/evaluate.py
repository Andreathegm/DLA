import torch
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from src.policynet import PolicyNet
from src.device import device


EVAL = False 


def record_video_evaluation(policy , env_name = "CartPole-v1",video_folder = "./videos",name_prefix = "final"):
    policy.eval()
    env_video = gym.make(env_name, render_mode="rgb_array")
    env_video = RecordVideo(env_video, video_folder= video_folder, name_prefix= name_prefix)
    obs, info = env_video.reset()
    done = False

    while not done:
        obs_tensor = torch.tensor(obs, dtype=torch.float32).to(device)
        action = policy(obs_tensor).argmax().item()
        obs, reward, terminated, truncated, info = env_video.step(action)
        done = terminated or truncated

    env_video.close()


if EVAL:

    print("Start Evaluation")
    env = gym.make("CartPole-v1")
    policy = PolicyNet(env).to(device)
    env.close()

    policy.load_state_dict(torch.load("weights/policy_final.pt"))
    print("policy loaded")

    record_video_evaluation(
        policy=policy,
        env_name="CartPole-v1",
        video_folder="./videos",
        name_prefix="eval_final"
    )

    print("Video saved in ./videos/")