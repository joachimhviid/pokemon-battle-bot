import numpy as np
import gym
import torch as torch 
import torch.nn as nn
import torch.optim as optim
import random
from gym import spaces
from collections import deque

class PokemonEnv(gym.Env):
    def __init__(self):
        super(PokemonEnv, self).__init__()
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=0, high=1, shape=(3,), dtype=np.float32)

        # Initialize the state of the environment
        self.reset()

    def reset(self):
        # Resets the battle to the initial state.
        self.agent_health = 1.0
        self.opponent_health = 1.0
        self.moves_availabile = 1.0 

        return np.array([self.agent_health, self.opponent_health, self.moves_availabile], dtype=np.float32)
    
    def steps(self, action):
        
        reward = 0
        done = False
        dmg = np.random.uniform(0.2, 0.4)
        if action == 0: #Attack Action
            self.opponent_health = max(0, self.opponent_health - dmg)
            reward = dmg * 2

        elif action == 1: #Item Action
            heal = np.random.uniform(0.1, 0.3)
            self.agent_health = min(1, self.agent_health + heal)
            reward = heal * 1.2

        elif action == 2: #Switch Action
            reward = (dmg * 2 )/4
        
        if self.opponent_health <= 0:
            reward += 30
            done = True
        
        if self.agent_health <= 0:
            reward -= 30
            done = True

        return np.array([self.agent_health, self.opponent_health, self.moves_availabile], dtype=np.float32), reward, done, {}

    def render(self):
        print(f"Agent Health: {self.agent_health:.2f} | Opponent Health: {self.opponent_health:.2f}")
        print(f"Moves Available: {self.moves_availabile}")


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
     



#Hyper Parameters and can fluctuate to suit a purpose
learning_rate = 0.001
gamma = 0.95 #Discount Factor
epsilon = 0.9 # Initial Exploration rate
epsilon_min = 0.01 
epsilon_decay = 0.8
batch_size = 32
memory_size = 1000

# TO Initialize env and model
env = PokemonEnv()
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

#Creating the DQN model and optimizer
model = DQN(state_size, action_size)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
loss_fn = nn.MSELoss() #Mean Squared Error Loss

memory = deque(maxlen=memory_size)


num_episodes = 1000
for episode in range(num_episodes):
    state = env.reset()
    state = torch.FloatTensor(state)
    total_reward = 0

    for t in range(600): # 600 is max steps per episode so change after need
        if random.random() < epsilon:
            action = env.action_space.sample() # Random action aka. Exploration
        else:
            with torch.no_grad():
                action = torch.argmax(model(state)).item() # Best Action aka Exploitation

        next_state, reward, done, t = env.steps(action)
        next_state = torch.FloatTensor(next_state)
        total_reward += reward

        memory.append((state, action, reward, next_state, done))

        if len(memory) > batch_size:
            minibatch = random.sample(memory, batch_size)
            for s, a, r, s_next, d in minibatch:
                target = r + (gamma * torch.max(model(s_next)).item() * (1 - d))
                predicted = model(s)[a]
                loss = loss_fn(predicted, torch.tensor(target))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
        state = next_state
        if done:
            break

        # Epsilon Decay exploration rate
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        if episode % 100 == 0:
            print(f"Episode: {episode}, Total Reward: {total_reward:.2f}, Epsilon: {epsilon:.3f}")


"""
# Load Trained Model
model.load_state_dict(torch.load("pokemon_ai_model.pth"))
model.eval()  # Set model to evaluation mode

# Run a battle
state = env.reset()
state = torch.FloatTensor(state)
done = False

while not done:
    with torch.no_grad():
        action = torch.argmax(model(state)).item()  # Choose best action
    state, reward, done, _ = env.step(action)
    state = torch.FloatTensor(state)
    env.render()  # Print battle status"
"""