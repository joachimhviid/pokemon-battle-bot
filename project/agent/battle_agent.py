import random
from collections import namedtuple
from typing import TYPE_CHECKING

import gymnasium as gym
import numpy as np
import torch

# from project import BattleEnv
from project.agent.dqn import DQN, ReplayMemory

if TYPE_CHECKING:
    from project.battle.battle_env import BattleEnv

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))


class BattleAgent:
    def __init__(self, env: 'BattleEnv',
                 learning_rate: float = 0.001,
                 gamma: float = 0.99,
                 epsilon_start: float = 1.0,
                 epsilon_end: float = 0.05,
                 epsilon_decay: float = 0.995,
                 memory_size: int = 10000,
                 batch_size: int = 64):
        self.env = env
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Calculate flattened observation space size
        sample_obs = env.observation_space.sample()
        flattened_size = self._flatten_observation(sample_obs).shape[0]

        # Initialize networks
        self.policy_net = DQN(flattened_size, env.action_space_size).to(self.device)
        self.target_net = DQN(flattened_size, env.action_space_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        # Initialize optimizer
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=learning_rate)

        # Initialize replay memory
        self.memory = ReplayMemory(memory_size)

        # Training parameters
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size

    def _flatten_observation(self, observation):
        # Convert dictionary observation to flat array
        flat_obs = []
        for key, value in observation.items():
            if isinstance(value, np.ndarray):
                flat_obs.extend(value.flatten())
            else:
                flat_obs.append(value)
        return np.array(flat_obs, dtype=np.float32)

    def choose_action(self, observation, action_mask: np.ndarray) -> int:
        # Add batch dimension to action_mask if it's not present
        if len(action_mask.shape) == 1:
            action_mask = action_mask[np.newaxis, :]  # Shape becomes [1, 14]

        if random.random() < self.epsilon:
            # Random choice from valid actions only
            valid_actions = np.where(action_mask[0])[0]
            if len(valid_actions) == 0:
                print(action_mask)
            # print(valid_actions)
            return np.random.choice(valid_actions)

        with torch.no_grad():
            state_tensor = torch.FloatTensor(self._flatten_observation(observation)).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_tensor)

            # Apply mask by setting invalid action Q-values to -inf
            masked_q_values = q_values.clone()
            masked_q_values[~torch.from_numpy(action_mask)] = float('-inf')

            return masked_q_values.argmax().item()

    def update_epsilon(self):
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def store_transition(self, state, action, next_state, reward, done):
        state = self._flatten_observation(state)
        next_state = self._flatten_observation(next_state)
        self.memory.append(Transition(
            torch.FloatTensor(state).to(self.device),
            torch.tensor([action]).to(self.device),
            torch.FloatTensor(next_state).to(self.device),
            torch.tensor([reward], dtype=torch.float32).to(self.device),
            torch.tensor([done], dtype=torch.bool).to(self.device),
        ))

    def train(self, action_mask: np.ndarray):
        if len(self.memory) < self.batch_size:
            return

        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        # Stack all tensors in the batch
        state_batch = torch.stack(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        next_state_batch = torch.stack(batch.next_state)
        done_mask = torch.cat(batch.done)

        # Compute Q(s_t, a)
        state_action_values = self.policy_net(state_batch).gather(1, action_batch.unsqueeze(1))

        # Initialize next_state_values with zeros
        next_state_values = torch.zeros(self.batch_size, device=self.device)

        # Compute V(s_{t+1}) for all non-final states at once
        with torch.no_grad():
            non_final_mask = ~done_mask
            if non_final_mask.any():
                non_final_next_states = next_state_batch[non_final_mask]
                next_q_values = self.target_net(non_final_next_states)

                # Expand action mask to match batch dimension
                action_mask_tensor = torch.from_numpy(action_mask).to(self.device)
                action_mask_tensor = action_mask_tensor.expand(next_q_values.size(0), -1)

                # Apply action mask to all next states
                next_q_values = next_q_values.masked_fill(~action_mask_tensor, float('-inf'))
                next_state_values[non_final_mask] = next_q_values.max(1)[0]

        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        # Compute loss
        loss = torch.nn.functional.smooth_l1_loss(
            state_action_values,
            expected_state_action_values.unsqueeze(1)
        )

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
