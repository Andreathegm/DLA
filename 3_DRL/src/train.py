import torch
import numpy as np
import gymnasium as gym
import os
from src.policynet import PolicyNet
from src.reinforce import reinforce
from src.device import device
from src.evaluate import record_video_evaluation
from src.plot import plot_results



def train(seed = 1234 ,env_string = "CartPole-v1",record_final_video = True,w_path = "weights"):

    torch.manual_seed(seed)
    np.random.seed(seed)


    env = gym.make(env_string)
    env.reset(seed=seed)

    policy = PolicyNet(env).to(device)


    w_eval_path = f"{w_path}/eval"
    os.makedirs(f"{w_eval_path}",exist_ok= True)


    running_rewards,eval_rewards,eval_lenghts = reinforce(policy=policy, env=env, num_episodes=1000,w_eval_path = w_eval_path)

    env.close()

    torch.save(policy.state_dict(), f"{w_path}/policy_final.pt")

    os.makedirs("numeric", exist_ok= True)
    np.save("numeric/running_rewards",running_rewards)
    np.save("numeric/eval_rewards",eval_rewards)
    np.save("numeric/eval_lenght",eval_lenghts)

    if record_final_video : 
        policy.eval()
        record_video_evaluation(policy = policy)
    
    return running_rewards,eval_rewards,eval_lenghts 