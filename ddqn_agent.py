# ddqn_agent.py

"""
DDQN Agent for CAV Cybersecurity Defense

This agent is designed to work with environments that have dynamic, modular action spaces,
such as cav_env_v3.py. It supports MultiDiscrete action spaces by flattening them into a 
single discrete output, allowing the agent to select optimal configurations (e.g., software
language versions) for each system module.

Features:
- Compatible with modular, multi-dimensional action spaces
- Learns optimal module-language pairings using Double DQN
- Fully supports MultiDiscrete to flat index mapping for seamless integration
- Includes epsilon-greedy policy, replay buffer, and soft target updates
"""


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque

# ===========================
# Neural Network Architecture
# ===========================
class DDQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DDQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# ===========================
# DDQN Agent Class
# ===========================
class DDQNAgent:
    def __init__(self, obs_dim, action_dims, gamma=0.99, lr=1e-3, batch_size=64, memory_size=10000):
        self.obs_dim = obs_dim
        self.action_dims = action_dims
        self.num_actions = int(np.prod(action_dims))

        self.gamma = gamma
        self.lr = lr
        self.batch_size = batch_size
        self.memory = deque(maxlen=memory_size)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy_net = DDQN(input_dim=obs_dim, output_dim=self.num_actions).to(self.device)
        self.target_net = DDQN(input_dim=obs_dim, output_dim=self.num_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)

        self.epsilon = 1.0
        self.epsilon_min = 0.02
        self.epsilon_decay = 0.995
        self.tau = 0.005

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randint(0, self.num_actions - 1)
        state_tensor = torch.FloatTensor(np.array(state)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)
        return torch.argmax(q_values, dim=1).item()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)

        states = torch.FloatTensor(np.array(states)).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        q_values = self.policy_net(states).gather(1, actions)
        next_actions = torch.argmax(self.policy_net(next_states), dim=1, keepdim=True)
        next_q_values = self.target_net(next_states).gather(1, next_actions)
        expected_q = rewards + (1 - dones) * self.gamma * next_q_values

        loss = F.mse_loss(q_values, expected_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        for target_param, local_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)

    def update_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path):
        torch.save(self.policy_net.state_dict(), path)
        print(f" Model saved to {path}")

    def load(self, path):
        self.policy_net.load_state_dict(torch.load(path))
        self.target_net.load_state_dict(self.policy_net.state_dict())
        print(f" Model loaded from {path}")
'''

# Write to file (save path)
