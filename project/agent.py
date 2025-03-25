import gym
import gym.envs

import numpy as np
import torch, torch.nn as nn, torch.optim as optim
import random
import asyncio

from gym import spaces
from collections import deque
from matplotlib import pyplot as plt
from dqn import DQN
from poke_env import Player
from poke_env.environment import Battle
from poke_env import AccountConfiguration, ServerConfiguration
from poke_env.player import RandomPlayer

class PokemonAgent(Player):
    def __init__(self, battle_format="gen9randombattle", server_configuration=None):
        super(PokemonAgent, self).__init__(battle_format=battle_format, server_configuration=server_configuration)
        
        # Define action/state space
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=0, high=1, shape=(2,), dtype=np.float32)
        
        # Initialize the model and memory
        self.model = DQN(self.observation_space.shape[0], self.action_space.n)
        self.gamma = 0.95
        self.epsilon= 0.8
        self.epsilon_decay = 0.99
        self.epsilon_min = 0.05
        self.learning_rate = 0.001
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        self.loss_fn = nn.MSELoss()
        self.memory = deque(maxlen=100000)
        
        self.training_errors = []
        self.win_log = []
        
        
    def embed_battle(self, battle: Battle) -> np.array:
        active = battle.active_pokemon
        opponent = battle.opponent_active_pokemon

        if active is None or opponent is None:
            return np.zeros(2)
        
        return np.array([
            active.current_hp_fraction, 
            opponent.current_hp_fraction
            ], dtype=np.float32)
    
    def choose_move(self, battle: Battle):
        state = self.embed_battle(battle)
        state_tensor = torch.FloatTensor(state)
        move_idx = 0
        
        if random.random() < self.epsilon:
            #Exploration
            available_moves = list(range(len(battle.available_moves)))
            print(available_moves)
            move_idx = random.choice(available_moves)
        else:
            #Exploitation
            with torch.no_grad():
                q_values = self.model(state_tensor)
                move_idx = torch.argmax(q_values).item()
        
        available_moves = battle.available_moves
        if move_idx >= len(available_moves):
            move_idx = 0 # Fallback to first move if move index is invalid
        
        return self.create_order(available_moves[move_idx])
    
    def train_experience(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

        if len(self.memory) < 32:
            return
        
        batch = random.sample(self.memory, 32)
        for s, a, r, s_next, d in batch:
            s_tensor = torch.FloatTensor(s)
            s_next_tensor = torch.FloatTensor(s_next)

            target = r + self.gamma * torch.max(self.model(s_next_tensor)).item() * (1 - d)
            prediction = self.model(s_tensor)[a]
            
            loss = self.loss_fn(prediction, torch.tensor(target))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


    def battle_end_callback(self, battle):
        result = 1 if battle.won else 0
        self.win_log.append(result)
    
    def save_model(self, path="/output.pth", name="Model-V1.0.0"):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path="/output.pth"):
        self.model.load_state_dict(torch.load(path))
        self.model.eval()



async def train_agent(agent, opponent, episodes=1000):
    rewards = []
    losses = []
    win_rates = []

    for episode in range(episodes):
        result = await agent.battle_against(opponent, n_battles=1)
        total_reward = 1 if agent.n_won_battle > 0 else 0
        rewards.append(total_reward)
        win_rate = np.mean(agent.win_log[-10:])
        win_rates.append(win_rate)
        if agent.training_errors:
            losses.append(np.mean(agent.training_errors))

        print(f"Episode: {episode+1}/{episodes} | Reward: {total_reward:.2f} | Win Rate: {win_rate:.2f} | Epsilon: {agent.epsilon:.2f}")

        agent.epsilon = max(agent.epsilon_min, agent.epsilon * agent.epsilon_decay)
    
    agent.save_model()
    return rewards, win_rates, losses

def plot_training(rewards, win_rates, losses):

    episodes = range(len(rewards))
    fig, axs = plt.subplots(1, 3, figsize=(18, 4))

    axs[0].plot(episodes, rewards, label="Rewards")
    axs[0].set_title("Episode Rewards")
    axs[0].set_xlabel("Episode")
    axs[0].set_ylabel("Reward")

    axs[1].plot(episodes, win_rates, label="Win Rate", color='green')
    axs[1].set_title("Rolling Win Rate (10)")
    axs[1].set_xlabel("Episode")
    axs[1].set_ylabel("Win %")

    if losses:
        axs[2].plot(episodes[:len(losses)], losses, label="Training Loss", color='red')
        axs[2].set_title("Average Training Loss")
        axs[2].set_xlabel("Episode")
        axs[2].set_ylabel("Loss")

    for ax in axs:
        ax.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    
    server_config = ServerConfiguration("ws://localhost:8000/showdown/websocket", None)
    opponent_config = AccountConfiguration("Random-Opponent", None)
#    agent_config = AccountConfiguration("AgentRL", None)
        
    agent = PokemonAgent(server_configuration=server_config)
    opponent = RandomPlayer(server_configuration=server_config, account_configuration=opponent_config)
    
    rewards, win_rates, losses = asyncio.run(train_agent(agent, opponent, episodes=1000))

    plot_training(rewards, win_rates, losses)