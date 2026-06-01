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
        self.actor = Actor(state_dim, action_dim).to(device)
        self.critic = Critic(state_dim).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(),lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(),lr=lr)
        self.memory = []

    def select_action(self,state):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            probs = self.actor(state)
        dist = Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action).item()

    def transaction(self,trans):
        self.memory.append(trans)

    def compute_advantages(self,rewards,dones,values,next_value):
        advantages = []
        gae = 0
        values = values.squeeze().cpu().numpy().tolist()
        values.append(next_value)

        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * values[t+1] * (1-dones[t]) - values[t]
            gae = delta + self.gamma * self.lambda_ * (1-dones[t]) * gae
            advantages.insert(0,gae)

        returns = [adv + val for adv,val in zip(advantages,values[:-1])]
        advantages = torch.FloatTensor(advantages).to(device)
        returns = torch.FloatTensor(returns).to(device)
        return advantages,returns

    def update(self):
        states, actions, old_log_probs, rewards, dones = zip(*self.memory)

        states = torch.FloatTensor(np.array(states)).to(device)
        actions = torch.LongTensor(actions).to(device)
        old_log_probs = torch.FloatTensor(old_log_probs).to(device)

        with torch.no_grad():
            values = self.critic(states).squeeze()
            next_value = 0

        advantages, returns = self.compute_advantages(rewards,dones,values,next_value)

        sampler = BatchSampler(SubsetRandomSampler(range(len(states))),batchsize,False)

        for epoch in range(epochs):
            for indices in sampler:
                batch_states = states[indices]
                batch_actions = actions[indices]
                batch_old_log_probs = old_log_probs[indices]
                batch_advantages = advantages[indices]
                batch_returns = returns[indices]

                probs = self.actor(batch_states)
                dist = Categorical(probs)
                new_log_probs = dist.log_prob(batch_actions)

                ratios = torch.exp(new_log_probs - batch_old_log_probs)

                val1 = ratios * batch_advantages
                val2 = torch.clamp(ratios,1-self.clips,1+self.clips) * batch_advantages

                policy_loss = -torch.min(val1,val2).mean()

                self.actor_optimizer.zero_grad()
                policy_loss.backward()
                self.actor_optimizer.step()

                values_pred = self.critic(batch_states).squeeze()
                value_loss = nn.MSELoss()(values_pred,batch_returns)

                self.critic_optimizer.zero_grad()
                value_loss.backward()
                self.critic_optimizer.step()

        self.memory = []
