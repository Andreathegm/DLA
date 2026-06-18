import torch
import numpy as np
import gymnasium as gym
import os
from src.policynet import PolicyNet
from src.reinforce import reinforce
from src.device import device
from src.evaluate import record_video_evaluation
from src.plot import plot_results



def train( env ,policy,lr = 1e-2, num_episodes = 1000,seed = 1234 ,baseline = None,record_final_video = True,w_path = "weights",v_path = "video"):

    torch.manual_seed(seed)
    np.random.seed(seed)

    env.reset(seed=seed)

    w_eval_path = f"{w_path}/eval"
    os.makedirs(f"{w_eval_path}",exist_ok= True)

    if baseline is not None :
        numeric_path = "numeric/baseline"
        print("Applying baseline")
        bs = baseline
    else:
        numeric_path = "numeric/standard"
        print("No baseline applied")
        bs = 0

    running_rewards,eval_rewards,eval_lenghts = reinforce(policy=policy,lr = lr ,baseline = bs, env=env, num_episodes=num_episodes,w_eval_path = w_eval_path)

    env.close()

    torch.save(policy.state_dict(), f"{w_path}/policy_final.pt")

    os.makedirs(numeric_path, exist_ok= True)
    np.save(f"{numeric_path}/running_rewards",running_rewards)
    np.save(f"{numeric_path}/eval_rewards",eval_rewards)
    np.save(f"{numeric_path}/eval_lenght",eval_lenghts)

    if record_final_video : 
        policy.eval()
        record_video_evaluation(policy = policy,video_folder = v_path)
    
    return running_rewards,eval_rewards,eval_lenghts 