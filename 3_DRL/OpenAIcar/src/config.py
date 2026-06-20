from dataclasses import dataclass
import os

@dataclass
class EnvConfig:
    gym_id: str
    num_envs: int = 4              ## vectorized enviroments - every env. should be i.i.d . Also done for reducing correlation btw sequential step
    seed: int = 1
    capture_video: bool = False
    run_name: str

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

    @property
    def batch_size(self) -> int:
        return int(4 * self.num_steps) 

    @property
    def minibatch_size(self) -> int:
        return int(self.batch_size // self.num_minibatches)