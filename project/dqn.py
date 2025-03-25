
import torch as torch 
import torch.nn as nn


class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = (nn.Linear(state_size, 64)) #Input layer -> Hidden layer
        self.fc2 = (nn.Linear(64, 64)) #Hidden layer -> Hidden layer
        self.fc3 = (nn.Linear(64, action_size)) #Hidden layer -> Output layer

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x 