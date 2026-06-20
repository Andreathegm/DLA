from dataclasses import dataclass
import argparse,time    

@dataclass
class EnvConfig:
    gym_id: str = "CarRacing-v3"
    num_envs: int = 4              ## vectorized enviroments - every env. should be i.i.d . Also done for reducing correlation btw sequential step
    seed: int = 1
    capture_video: bool = False
    run_name: str = "first_run"

@dataclass
class PPOConfig:
    total_timesteps: int = 25000   ## number of TOTAL env step
    learning_rate: float = 2.5e-4
    num_envs : int = 4
    num_steps: int = 128           ## 1) T - number of step to get a rollout from ONE env. 2)Total size of collected batch is (num_env*num_step)
    anneal_lr: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95       ## smoothing paramter for Generalized Advanteges estimation
    num_minibatches: int = 4       ## How many chunks to split the collected rollout batch into during optimization.
    update_epochs: int = 4         ## How many times to iterate over the same collected rollout batch to update the neural networks.
    clip_coef: float = 0.2         ## r(theta) be the probability ratio of taking an action under the new policy vs the old policy. simply the parameter caps the ratio btw 1+-clip_coef
    clip_vloss: bool = True        ## Toggles whether to clip the value network updates similarly to the policy updates.
    ent_coef: float = 0.01         ## Let S be the entropy of the policy , ent_coef is the entropy coefficient -->  -ent_coef*S to maximize entropy so we try to prevent overconfidence
    vf_coef: float = 0.5           ## Value function coefficeint, to scale the value loss function
    max_grad_norm: float = 0.5     ## max L2 norm , so the gradient are scaled down 
    target_kl: float = None        ## KL divergence      
    norm_adv: bool = True          ## Toggles advantage normalization at batch level
    checkpoint_freq: int = 50

    @property
    def batch_size(self) -> int:
        return int(self.num_envs * self.num_steps) 

    @property
    def minibatch_size(self) -> int:
        return int(self.batch_size // self.num_minibatches)

def parse_args():
    parser = argparse.ArgumentParser(description="PPO Trainer for OpenAI Car")
    
    parser.add_argument("--gym-id", type=str, default="CarRacing-v3")
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--total-timesteps", type=int, default=1000000)
    parser.add_argument("--learning-rate", type=float, default=2.5e-4)
    parser.add_argument("--num-steps", type=int, default=128)
    parser.add_argument("--checkpoint-freq", type=int, default=50)
    
    args = parser.parse_args()
    
    run_name = f"{args.gym_id}__seed_{args.seed}__{int(time.time())}"
    
    env_cfg = EnvConfig(
        gym_id=args.gym_id,
        num_envs=args.num_envs,
        seed=args.seed,
        run_name=run_name
    )
    
    ppo_cfg = PPOConfig(
        num_envs = args.num_envs,
        total_timesteps=args.total_timesteps,
        learning_rate=args.learning_rate,
        num_steps=args.num_steps,
        checkpoint_freq=args.checkpoint_freq
    )
    
    return env_cfg, ppo_cfg