import gymnasium as gym
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import random
import numpy as np
import time

from src.config import EnvConfig, PPOConfig
from src.buffer import RolloutBuffer
from src.models import DiscreteAgent
from src.trainer import PPOTrainer
from gymnasium.wrappers import GrayScaleObservation, FrameStack, ResizeObservation

def make_env(gym_id, seed, idx, capture_video, run_name):
    def thunk():

        env = gym.make(gym_id, continuous=False) ## set continuous = false
        
        env = gym.wrappers.RecordEpisodeStatistics(env)
        
        env = GrayScaleObservation(env, keep_dim=False) 
        
        env = ResizeObservation(env, (84, 84)) 
        
        env = FrameStack(env, 4)                ## Output shape: (4, 84, 84)
        
        if capture_video and idx == 0:
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
            
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env
    return thunk

def main():
    env_cfg = EnvConfig()
    ppo_cfg = PPOConfig()
    
    random.seed(env_cfg.seed)
    np.random.seed(env_cfg.seed)
    torch.manual_seed(env_cfg.seed)
    torch.backends.cudnn.deterministic = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_name = f"{env_cfg.gym_id}__{env_cfg.seed}__{int(time.time())}"
    writer = SummaryWriter(f"runs/{run_name}")

    envs = gym.vector.SyncVectorEnv(
        [make_env(env_cfg.gym_id, env_cfg.seed + i, i, env_cfg.capture_video, run_name) 
         for i in range(env_cfg.num_envs)]
    )

    obs_shape = envs.single_observation_space.shape
    action_shape = envs.single_action_space.shape
    action_dim = envs.single_action_space.n

    agent = DiscreteAgent(obs_shape, action_dim).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=ppo_cfg.learning_rate, eps=1e-5)
    
    buffer = RolloutBuffer(
        ppo_cfg.num_steps, env_cfg.num_envs, obs_shape, action_shape, device
    )

    trainer = PPOTrainer(agent, optimizer, ppo_cfg, envs, buffer, device, writer)
    trainer.train()

    envs.close()
    writer.close()

if __name__ == "__main__":
    main()