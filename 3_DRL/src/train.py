import torch
import numpy as np
import os
from src.reinforce import reinforce
from src.evaluate import record_video_evaluation



def train( env ,policy,lr,lr_vnet,num_episodes,seed,baseline,w_path,v_path,n_path,record_final_video = True,loss_op = "mean"):

    print(f"started trainig with lr = {lr}")
    w_eval_path = f"{w_path}/eval"
    os.makedirs(w_eval_path,exist_ok=True)

    torch.manual_seed(seed)
    np.random.seed(seed)

    env.reset(seed=seed)


    if baseline is not None :
        print("Applying baseline")
    else:
        print("No baseline applied")


    running_rewards,eval_rewards,eval_lenghts = reinforce(policy=policy,lr = lr ,lr_vnet= lr_vnet, loss_op = loss_op ,baseline = baseline, env=env, num_episodes=num_episodes,w_eval_path = w_eval_path)

    env.close()

    torch.save(policy.state_dict(), f"{w_path}/policy_final.pt")

    ### save numeric results
    np.save(f"{n_path}/running_rewards",running_rewards)
    np.save(f"{n_path}/eval_rewards",eval_rewards)
    np.save(f"{n_path}/eval_lenght",eval_lenghts)


    ### record final video
    if record_final_video : 
        policy.eval()
        record_video_evaluation(policy = policy,video_folder = v_path)

    return running_rewards,eval_rewards,eval_lenghts 