from collections import deque
import random

import torch


class DQN(torch.nn.Module):
    def __init__(self, n_observations, n_actions):
        super().__init__()
        self.fc1 = torch.nn.Linear(n_observations, 512)
        self.fc2 = torch.nn.Linear(512, 256)
        self.out = torch.nn.Linear(256, n_actions)

    def forward(self, x):
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        return self.out(x)


class ReplayMemory():
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def append(self, transition):
        self.memory.append(transition)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
