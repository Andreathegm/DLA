import torch
import numpy as np
import os
from src.reinforce import reinforce
from src.evaluate import record_video_evaluation



def train( env ,policy,loss_op , lr = 1e-2,num_episodes = 1000,seed = 1234 ,baseline = None,record_final_video = True,w_path = "weights",v_path = "video"):

    print(f"started trainig with lr = {lr}")

    torch.manual_seed(seed)
    np.random.seed(seed)

    env.reset(seed=seed)


    if baseline is not None :
        b = "baseline"
        w_final_path = f"{w_path}/{b}"
        w_eval_path = f"{w_final_path}/eval"
        os.makedirs(f"{w_eval_path}",exist_ok= True)
        numeric_path = f"numeric/{b}"
        print("Applying baseline")
    else:
        s = "standard"
        w_final_path = f"{w_path}/{s}"
        w_eval_path = f"{w_final_path}/eval"
        os.makedirs(f"{w_eval_path}",exist_ok= True)
        numeric_path = f"numeric/{s}"
        print("No baseline applied")

    running_rewards,eval_rewards,eval_lenghts = reinforce(policy=policy,lr = lr , loss_op = loss_op ,baseline = baseline, env=env, num_episodes=num_episodes,w_eval_path = w_eval_path)

    env.close()

    torch.save(policy.state_dict(), f"{w_final_path}/policy_final.pt")
    os.makedirs(numeric_path, exist_ok= True)
    np.save(f"{numeric_path}/running_rewards",running_rewards)
    np.save(f"{numeric_path}/eval_rewards",eval_rewards)
    np.save(f"{numeric_path}/eval_lenght",eval_lenghts)

    if record_final_video : 
        policy.eval()
        record_video_evaluation(policy = policy,video_folder = v_path)
    return running_rewards,eval_rewards,eval_lenghts 