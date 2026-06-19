import torch.nn as nn
import torch.nn.functional as F

class ValueNet(nn.Module):
    def __init__(self,env):
        super().__init__()
        self.fc1 = nn.Linear(env.observation_space.shape[0], 128)
        self.fc2 = nn.Linear(128, env.action_space.n)
    
    def forward(self,v) :
        v = F.relu(self.fc1(v))
        v = F.softmax(self.fc2(v))
        return v

