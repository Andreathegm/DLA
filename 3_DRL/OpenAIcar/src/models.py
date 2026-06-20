import numpy as np
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)        ## so the weight colummns has dot product equal to zero
    torch.nn.init.constant_(layer.bias, bias_const)     ## fill in the tensor with the value of bias_const
    return layer

class CNNDiscreteAgent(nn.Module):
    def __init__(self, obs_shape, action_dim):
        super().__init__()
        channels, _, _ = obs_shape                      ## obs_shape (stack_frames, hight , width) i.e. (4, 96, 96) for example

        
        self.network = nn.Sequential(                   ## Feature Extractor (CNN)

            nn.Conv2d(channels, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Flatten(),                               ## to be passed to the MLP heads
        )

        with torch.no_grad():
            n_flatten = self.network(torch.zeros(1, *obs_shape)).shape[1]
                                                         ## used to calculate dynamicly the output of the convulation with the formula 
                                                         ## o = floor(I - K + 2P  / S)

        # 2. Critic Head Value_function
        self.critic = nn.Sequential(
            layer_init(nn.Linear(n_flatten, 256)),
            nn.ReLU(),
            layer_init(nn.Linear(256, 1), std=1.0),      ## TODO : wide range why ? 
        ) 

        # 3. Actor Head  
        self.actor = nn.Sequential(
            layer_init(nn.Linear(n_flatten, 256)),
            nn.ReLU(),
            layer_init(nn.Linear(256, action_dim), std=0.01),  ##TODO More exploration so at the stard we have uniform distribution kinda
        )

    def get_value(self, x):
                                                     ##TODO x should be normalized before entering here because 
                                                     ##  neural networks expect inputs roughly in the $[-1, 1]$ or $[0, 1]$ range to keep activations stable.
        x = x.float()/255
        hidden = self.network(x)
        return self.critic(hidden)

    def get_action_and_value(self, x, action=None):  ## TODO : why passing the action in the first place
        
        x = x.float()/255                            ## We convert to float because in this way even if the tensor will contain int values we convert them before dividing them


        ## first we get our features
        hidden = self.network(x)
        
        ## we run the policy_net (actor) 
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        
        if action is None:            
            action = probs.sample()
            
        # we also run value_net (critic) and return...
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden)