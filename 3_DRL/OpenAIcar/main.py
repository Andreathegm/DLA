import gymnasium as gym
import torch
import torch.optim as optim
import random
import numpy as np
import os
import sys

from gymnasium.wrappers import GrayscaleObservation,ResizeObservation,FrameStackObservation
from torch.utils.tensorboard import SummaryWriter
from src.config import parse_args
from src.buffer import RolloutBuffer
from src.models import CNNDiscreteAgent
from src.trainer import PPOTrainer


def make_env(gym_id, seed, idx, capture_video, run_name):
    def thunk():
        
        env = gym.make(gym_id, continuous=False) 
        obs,info = env.reset(seed=seed)
        print("Observation shape :",obs.shape,info)                                  
        env.action_space.seed(seed)
        env.observation_space.seed(seed)

        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = GrayscaleObservation(env, keep_dim=False)
        env = ResizeObservation(env, (84, 84)) 
        env = FrameStackObservation(env, 4)
        obs,info = env.reset(seed=seed)
        print("Observation shape :",obs.shape,info)
        if capture_video and idx == 0:
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")

        return env
    return thunk

def main():
    env_cfg, ppo_cfg = parse_args()
    
    run_dir = f"runs/{env_cfg.run_name}"
    os.makedirs(run_dir, exist_ok=True)
    
    command_log_path = os.path.join(run_dir, "command.txt")
    with open(command_log_path, "w") as f:
        f.write("python " + " ".join(sys.argv) + "\n")
    
    random.seed(env_cfg.seed)
    np.random.seed(env_cfg.seed)
    torch.manual_seed(env_cfg.seed)
    torch.backends.cudnn.deterministic = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    writer = SummaryWriter(run_dir)

    envs = gym.vector.SyncVectorEnv(
        [make_env(env_cfg.gym_id, env_cfg.seed + i, i, env_cfg.capture_video, env_cfg.run_name) 
         for i in range(env_cfg.num_envs)]
    )

    obs_shape = envs.single_observation_space.shape
    action_dim = envs.single_action_space.n

    agent = CNNDiscreteAgent(obs_shape, action_dim).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=ppo_cfg.learning_rate, eps=1e-5)
    
    buffer = RolloutBuffer(
        ppo_cfg.num_steps, env_cfg.num_envs, obs_shape, (), device
    )

    trainer = PPOTrainer(agent, optimizer, ppo_cfg, envs, buffer, device, writer, run_dir)
    trainer.train()

    envs.close()
    writer.close()

if __name__ == "__main__":
    main()