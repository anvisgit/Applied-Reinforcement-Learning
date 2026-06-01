import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from torch.distributions import Categorical
from torch.utils.data import BatchSampler, SubsetRandomSampler

episodes=2000
gamma=0.96
lambdav=0.95 #ppo
clips=0.2 
lr=0.001

epochs=10
batchsize=64
timestamp=1000
loginterval=20

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()

        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.action_head = nn.Linear(128, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return torch.softmax(self.action_head(x), dim=-1)


class Critic(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.value_head = nn.Linear(128, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.value_head(x)


class PPO:
  def __init__(self,state_dim, action_dim, lr, gamma, lambdav, clips):
        self.gamma = gamma
        self.lambda_ = lambdav
        self.clips = clips
        self.actor=Actor(state_dim, action_dim).to(device)
        self.critic = Critic(state_dim)).to(device)

        self.optimizer=optim.Adam(self.actor.parameters(), lr=lr)
        self.optimizer=optim.Adam(self.critic.parameters(), lr=lr)
        self.memory = []
     


