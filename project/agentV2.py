import numpy as np
import math, time, random, inspect, os, asyncio
from collections import deque, namedtuple

import gymnasium as gym
from gymnasium import spaces


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from poke_env.environment import observation
from poke_env import Player, AccountConfiguration, ServerConfiguration
from poke_env.environment import AbstractBattle
from poke_env.player import Gen9EnvSinglePlayer, RandomPlayer, MaxBasePowerPlayer, ObsType
from poke_env.environment.pokemon_type import PokemonType

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

N_BATTLES = 10       # Total number of battles to train for
BATCH_SIZE = 128       # Number of experiences to sample from buffer for learning
GAMMA = 0.99           # Discount factor for future rewards
EPSILON_START = 0.9        # Starting exploration rate
EPSILON_END = 0.05         # Minimum exploration rate
EPSILON_DECAY = 10000      # How fast the exploration rate decays (higher = slower decay)
TAU = 0.005            # Soft update factor for target network
LR = 1e-4              # Learning rate for the optimizer
BUFFER_SIZE = 20000    # Max size of the replay buffer
TARGET_UPDATE_FREQ = 5 # How often (in episodes/battles) to update target network weights
LOG_FREQ = 50 # How often (in episodes/battles) to print progress
DEBUG_STEPS = 5

NUM_TYPES = 18 # Number of types in the game
STATE_SIZE = 2 + 2 + (NUM_TYPES * 2) + (NUM_TYPES * 2) + 7 + 7
ACTION_SPACE_SIZE = 10 # Max number of Actions (moves + switches) available in a battle

BATTLE_FORMAT = "gen9randombattle" # Battle format for the environment
SERVER_CONF = ServerConfiguration("ws://localhost:8000/showdown/websocket", None)
OPP_ACC_CONF = AccountConfiguration(f"Random-{random.randint(0, 10000)}", None)
AGENT_ACC_CONF = AccountConfiguration("DQNAgent", None)
# --- Helper Function to Print Args/Kwargs ---
def print_args(func_name, *args, **kwargs):
    print(f"--- Entering {func_name} ---")
    if args:
        print(f"  Args: {args}")
    if kwargs:
        # Try to get arg names if possible (useful for __init__)
        try:
            sig = inspect.signature(globals()[func_name.split('.')[0]].__init__) # Crude way to get signature
            bound_args = sig.bind_partial(*args, **kwargs)
            bound_args.apply_defaults()
            print("  Keyword Args (Bound):")
            for name, value in bound_args.arguments.items():
                 # Skip 'self' if present
                 if name == 'self':
                     continue
                 # Truncate long values for cleaner printing
                 value_repr = repr(value)
                 if len(value_repr) > 100:
                     value_repr = value_repr[:97] + "..."
                 print(f"    {name}: {value_repr}")
        except Exception: # Fallback if inspection fails
            print(f"  Keyword Args: {kwargs}")
    print(f"--- Finished Args for {func_name} ---")

class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        # n_observations must match STATE_SIZE (88)
        # n_actions must match ACTION_SPACE_SIZE (10)
        print_args("DQN.__init__", n_observations=n_observations, n_actions=n_actions)
        super(DQN, self).__init__()
        # ***** THIS MUST USE THE CORRECT n_observations (STATE_SIZE) *****
        self.layer1 = nn.Linear(n_observations, 256)
        self.layer2 = nn.Linear(256, 128)
        self.layer3 = nn.Linear(128, n_actions)
        print(f"  DQN Initialized: Input Size={n_observations}, Output Size={n_actions}") # Verify input size
        print(f"  DQN Layers: Linear({n_observations}, 256) -> ReLU -> Linear(256, 128) -> ReLU -> Linear(128, {n_actions})")

    def forward(self, x):
        # Ensure input is float32
        if x.dtype != torch.float32:
             x = x.to(torch.float32)
        # Add a check for input shape just before the layer call
        if x.shape[1] != self.layer1.in_features:
             raise RuntimeError(f"Shape mismatch in DQN forward! Input shape {x.shape} does not match layer1 expected input features {self.layer1.in_features}")
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

class MyAgent(Gen9EnvSinglePlayer):
    def __init__(
      self,
      *args,
      **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.type_map = {t.name.lower():  i for i, t in enumerate(PokemonType) if i < NUM_TYPES}
        if len(self.type_map) != NUM_TYPES:
             print(f"Warning: Expected {NUM_TYPES} types in PokemonType enum, but found {len(self.type_map)}. STATE_SIZE might be incorrect.")
        
        
    def embed_battle(self, battle: AbstractBattle) -> np.ndarray:

        my_active = battle.active_pokemon
        opp_active = battle.opponent_active_pokemon
        my_hp = my_active.current_hp_fraction if my_active else 0
        opp_hp = opp_active.current_hp_fraction if opp_active else 0
        my_team_size = len([p for p in battle.team.values() if p.fainted is False]) / 6.0
        opp_team_size = len([p for p in battle.opponent_team.values() if p.fainted is False]) / 6.0
        
        my_types_vector = np.zeros(NUM_TYPES * 2) # 36 elements
        if my_active:
            if my_active.type_1:
                type_1_name = my_active.type_1.name.lower()
                if type_1_name in self.type_map:
                    my_types_vector[self.type_map[type_1_name]] = 1
            if my_active.type_2:
                type_2_name = my_active.type_2.name.lower()
                if type_2_name in self.type_map:
                     # Use second half of the vector for the second type
                    my_types_vector[self.type_map[type_2_name] + NUM_TYPES] = 1

        opp_types_vector = np.zeros(NUM_TYPES * 2) # 36 elements
        if opp_active:
            if opp_active.type_1:
                type_1_name = opp_active.type_1.name.lower()
                if type_1_name in self.type_map:
                    opp_types_vector[self.type_map[type_1_name]] = 1
            if opp_active.type_2:
                type_2_name = opp_active.type_2.name.lower()
                if type_2_name in self.type_map:
                    # Use second half of the vector for the second type
                    opp_types_vector[self.type_map[type_2_name] + NUM_TYPES] = 1


        status_map = {None: 0, "paralysis": 1, "burn": 2, "freeze": 3, "poison": 4, "sleep": 5, "Toxic": 6}
        my_status_vector = np.zeros(7)
        opp_status_vector = np.zeros(7)
        my_status = my_active.status.name if my_active and my_active.status else None
        opp_status = opp_active.status.name if opp_active and opp_active.status else None
        my_status_vector[status_map.get(my_status, 0)] = 1
        opp_status_vector[status_map.get(opp_status, 0)] = 1
        
        # Concatenate features
        state = np.concatenate([
            np.array([my_hp, opp_hp], dtype=np.float32),                     # 2 elements
            np.array([my_team_size, opp_team_size], dtype=np.float32),       # 2 elements
            my_types_vector.astype(np.float32),                              # 36 elements
            opp_types_vector.astype(np.float32),                             # 36 elements
            my_status_vector.astype(np.float32),                             # 7 elements
            opp_status_vector.astype(np.float32)                             # 7 elements
        ]) # Total = 2+2+36+36+7+7 = 88 elements

        global STATE_SIZE
        if state.shape[0] != STATE_SIZE:
            print(f"\n!!! CRITICAL WARNING: State size mismatch! Expected {STATE_SIZE}, Got {state.shape[0]}. !!!")
            print(f"!!! FEATURES: my_hp={my_hp:.2f}, opp_hp={opp_hp:.2f}, my_team={my_team_size:.2f}, opp_team={opp_team_size:.2f}")
            print(f"!!!          my_types={my_types_vector.sum()}, opp_types={opp_types_vector.sum()}, my_status={my_status_vector.sum()}, opp_status={opp_status_vector.sum()}")
            print(f"!!! You MUST adjust STATE_SIZE in the script header to {state.shape[0]} and restart. !!!\n")
            # Adjusting dynamically can be risky, manual fix is better.
            # STATE_SIZE = state.shape[0] # Uncomment with caution

        return state
        
    def calc_reward(self, battle) -> float:
        
        if battle.won:
            return 10.0
        elif battle.lost:
            return -10.0
        else: return 0.0

    def action_to_move(self, action, battle: AbstractBattle):
        if action < len(battle.available_moves):
            return self.create_order(battle.available_moves[action])
        elif action < len(battle.available_moves) + len(battle.available_switches):
            switch_index = action - len(battle.available_moves)
            return self.create_order(battle.available_switches[switch_index])
        else:
            print(f"Warning: Action index {action} out of bounds. Choosing random valid action.")
            return self.choose_random_move(battle)
    
    # Implementation of the abstract method required by OpenAIGymEnv
    def describe_embedding(self) -> spaces.Space:
        """
        Returns the description of the embedding space (observation space).
        It must return a Gymnasium Space object.
        """
        # All values in our embedding are floats between 0.0 and 1.0
        low = np.zeros(STATE_SIZE, dtype=np.float32)
        high = np.ones(STATE_SIZE, dtype=np.float32)
        print(f"--- Describing Embedding Space ---")
        print(f"  Shape: ({STATE_SIZE},)")
        print(f"  Data Type: np.float32")
        print(f"  Low Bound: 0.0 (for all elements)")
        print(f"  High Bound: 1.0 (for all elements)")
        print(f"---------------------------------")
        return spaces.Box(low=low, high=high, shape=(STATE_SIZE,), dtype=np.float32)

# --- 3. Replay Buffer ---
Experience = namedtuple('Experience', ('state', 'action', 'reward', 'next_state', 'done'))

class ReplayBuffer:
    def __init__(self, capacity):
        print_args("ReplayBuffer.__init__", capacity=capacity)
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        state, action, reward, next_state, done = args
        if isinstance(state, torch.Tensor): state = state.cpu().numpy()
        if next_state is not None and isinstance(next_state, torch.Tensor): next_state = next_state.cpu().numpy()
        # Add a check for state size before pushing
        if state is not None and state.shape[0] != STATE_SIZE:
             print(f"ERROR: Pushing state with wrong size {state.shape} to buffer! Expected {STATE_SIZE}.")
             return # Avoid adding bad data
        if next_state is not None and next_state.shape[0] != STATE_SIZE:
             print(f"ERROR: Pushing next_state with wrong size {next_state.shape} to buffer! Expected {STATE_SIZE}.")
             return # Avoid adding bad data

        self.memory.append(Experience(state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)
    

class DQNAgent:
    def __init__(self, state_dim, action_dim):
        print_args("DQNAgent.__init__", state_dim=state_dim, action_dim=action_dim)
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.policy_net = DQN(state_dim, action_dim).to(DEVICE)
        print("DQNAgent: policy network created.")
        self.target_net = DQN(state_dim, action_dim).to(DEVICE)
        print("DQNAgent: target network created.")
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        print("DQNAgent: target network weights loaded with policy network and set to eval mode.")

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LR, amsgrad=True)
        print(f"DQNAgent: Optimizer AdamW created with lr={LR}, amsgrad=True.")
        self.replay_buffer = ReplayBuffer(BUFFER_SIZE)
        print(f"DQNAgent: Replay buffer created with capacity={BUFFER_SIZE}.")
        self.steps_done = 0
        print("DQNAgent: initialization complete.")

    def select_action(self, state, available_action_indencies):
        sample = random.random()
        epsilon_threshold = EPSILON_END + (EPSILON_START - EPSILON_END) * \
            math.exp(-1.0 * self.steps_done / EPSILON_DECAY)
        
        self.steps_done += 1

        if not available_action_indencies:
            print("Warning: No available actions. Returning random action.")
            return torch.tensor([[0]], device=DEVICE, dtype=torch.long)
        
        if sample > epsilon_threshold:
            with torch.no_grad():
                state_tensor = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
                print(f"DEBUG: state_tensor shape before policy_net call: {state_tensor.shape}")
                # Check network expected input size
                expected_features = self.policy_net.layer1.in_features
                print(f"DEBUG: policy_net expects input features: {expected_features}")
                if state_tensor.shape[1] != expected_features:
                     print(f"ERROR: Shape mismatch detected before calling policy_net!")
                     # Optionally raise error here too
                     # raise ValueError(f"State tensor shape {state_tensor.shape} does not match policy_net expected features {expected_features}")


                q_values = self.policy_net(state_tensor)

                mask = torch.full_like(q_values, -float("inf"))
                valid_indencies = torch.tensor(available_action_indencies, device=DEVICE, dtype=torch.long)
                
                if valid_indices_tensor.numel() > 0:
                     valid_indices_tensor = valid_indices_tensor[valid_indices_tensor < q_values.shape[1]]
                     if valid_indices_tensor.numel() > 0:
                         mask[0, valid_indices_tensor] = q_values[0, valid_indices_tensor]

                if torch.isinf(mask).all():
                    print("Warning: Masking resulted in all -inf Q-values during exploitation. Choosing random valid action.")
                    action_idx = random.choice(available_action_indencies)
                    return torch.tensor([[action_idx]], device=DEVICE, dtype=torch.long)
                else:
                    best_action_idx = mask.argmax(dim=1)
                    return best_action_idx.view(1, 1)
        else:
            action_index = random.choice(available_action_indencies)
            return torch.tensor([[action_index]], device=DEVICE, dtype=torch.long)
        
    def learn(self):
        if len(self.replay_buffer) < BATCH_SIZE:
            return None

        experiences = self.replay_buffer.sample(BATCH_SIZE)
        batch = Experience(*zip(*experiences))
        
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), 
                                      device=DEVICE, dtype=torch.bool)
        non_final_next_states_list = [torch.tensor(s, device=DEVICE, dtype=torch.float32) 
                                     for s in batch.next_state if s is not None]
        
        if non_final_next_states_list:
            np_states = np.array(non_final_next_states_list)
            if np_states.shape[1] != self.state_dim:
                print(f"ERROR: Non-final next states have wrong dimension {np_states.shape[1]} in learn! Expected {self.state_dim}")
                # Handle error: skip batch? return? raise?
                return None # Skip learning for this batch
            non_final_next_states = torch.tensor(np_states, device=DEVICE, dtype=torch.float32)
        else:
            non_final_next_states = torch.empty((0, self.state_dim), device=DEVICE, dtype=torch.float32)

        state_batch_np = np.array(batch.state)
        if state_batch_np.shape[1] != self.state_dim:
            print(f"ERROR: State batch has wrong dimension {state_batch_np.shape[1]} in learn! Expected {self.state_dim}")
            return None # Skip learning for this batch
        state_batch = torch.tensor(state_batch_np, dtype=torch.float32, device=DEVICE)
        

        action_batch = torch.tensor(batch.action, device=DEVICE, dtype=torch.long).unsqueeze(1)
        reward_batch = torch.tensor(batch.reward, device=DEVICE, dtype=torch.float32).unsqueeze(1)

        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(BATCH_SIZE, device=DEVICE)
        if non_final_next_states.shape[0] > 0:
            with torch.no_grad():
                next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0]
        
        expected_state_action_values = (next_state_values.unsqueeze(1) * GAMMA) + reward_batch

        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

        return loss.item()
    
    def update_target_network(self):
        target_net_state_dict = self.target_net.state_dict()
        policcy_net_state_dict = self.policy_net.state_dict()
        for key in policcy_net_state_dict:
            target_net_state_dict[key] = policcy_net_state_dict[key]*TAU + target_net_state_dict[key] * (1-TAU)
        self.target_net.load_state_dict(target_net_state_dict)


async def train_agent(n_battles_to_run):
    print(f"\n--- Starting Training Function: train_agent ---")
    print(f"  n_battles_to_run: {n_battles_to_run}")
    start_time = time.time()

    # --- Setup Environment and Players ---
    opponent = RandomPlayer(account_configuration=OPP_ACC_CONF, 
                            battle_format=BATTLE_FORMAT,
                            server_configuration=SERVER_CONF)
    print(f"  Opponent Player ({type(opponent).__name__}) Config:")
    print(f"    Battle Format: {BATTLE_FORMAT}")

    env_player = MyAgent(
        opponent=opponent,
        account_configuration=AGENT_ACC_CONF,       
        battle_format=BATTLE_FORMAT,
        log_level=20, # Optional but here for info
        server_configuration=SERVER_CONF
    )

    print(f"  RL Player ({type(env_player).__name__}) set up.")

    _ = env_player.describe_embedding()
    print(f"  Environment embedding space described.")

    env = env_player # Use the player directly as the environment
    print(f"  Environment set to RL Player instance.")

    # --- Setup Agent ---
    print(f"\n--- Initializing DQNAgent ---")
    agent = DQNAgent(STATE_SIZE, ACTION_SPACE_SIZE)
    print(f"--- DQNAgent Initialized ---\n")

    print(f"--- Starting Training Loop for {n_battles_to_run} Battles ---")
    wins = 0
    total_reward = 0
    recent_rewards = deque(maxlen=LOG_FREQ)
    #recent_losses = deque(maxlen=LOG_FREQ)
    #total_losses = 0.0

    for episode in range(1, n_battles_to_run + 1):
        state_dict, info = env.reset()
        state = env.embed_battle(env.current_battle)
        if episode == 1:
            print(f"  Initial state shape from embed_battle: {state.shape}")
        
        done = False
        episode_reward = 0.0
        episode_loss = 0.0
        steps = 0
        turn = 0

        while not done:
            turn += 1
            current_action_space_size = env.action_space.n
            available_action_indencies = list(range(current_action_space_size))

            # Ensure state is numpy array before passing to agent
            if state is None:
                 print(f"ERROR: State is None before selecting action in Ep {episode}, Turn {turn}. Breaking episode.")
                 break # Should not happen if done logic is correct
            if isinstance(state, torch.Tensor):
                state = state.cpu().numpy()
            # Add shape check before calling select_action
            if state.shape[0] != STATE_SIZE:
                 print(f"ERROR: State shape is {state.shape} before select_action! Expected ({STATE_SIZE},). Breaking episode.")
                 break

            action_tensor = agent.select_action(state, available_action_indencies)
            action = action_tensor.item()

            
            if action >= current_action_space_size:
                if turn <= DEBUG_STEPS: # Only print warning for early steps
                     print(f"  [Ep {episode}, Turn {turn}] Warning: Agent chose invalid action {action} (>= {current_action_space_size}). Choosing random.")
                action = random.choice(list(range(current_action_space_size)))
                action_tensor = torch.tensor([[action]], device=DEVICE, dtype=torch.long)

            next_state_dict, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            next_state = env.embed_battle(env.current_battle) if not done else None

            # Print step details only for the first few steps
            if turn <= DEBUG_STEPS:
                print(f"  [Ep {episode}, Turn {turn}] Action: {action}, Reward: {reward:.2f}, Done: {done}"
                      f", Available Actions: {current_action_space_size}")
                if next_state is not None:
                     print(f"     Next State (Embed): {np.round(next_state[:4], 2)}...") # Print first few features
                else:
                     print("     Next State: None (Episode Ended)")

            agent.replay_buffer.push(state, action, reward, next_state, done)

            state = next_state
            loss = agent.learn()
            if loss is not None:
                episode_loss += loss
                steps += 1

            episode_reward += reward

            if done:
                break
        
        total_reward += episode_reward
        recent_rewards.append(episode_reward)
        if env.current_battle.won:
            wins += 1
        
        if episode % TARGET_UPDATE_FREQ == 0:
            agent.update_target_network()
        
        if episode % LOG_FREQ == 0:
            avg_loss = (episode_loss / steps) if steps > 0 else 0
            avg_reward = sum(recent_rewards) / len(recent_rewards)
            win_rate = wins / LOG_FREQ
            elasped_time = time.time() - start_time
            current_epsilon = EPSILON_END + (EPSILON_START - EPSILON_END) * \
                math.exp(-1.0 * agent.steps_done / EPSILON_DECAY)
            print(f"\n--- Episode Log ---")
            print(f"  Episode: {episode}/{n_battles_to_run}")
            print(f"  Avg Reward (Last {LOG_FREQ}): {avg_reward:.3f}")
            print(f"  Avg Loss (This Episode): {avg_loss:.5f}")
            print(f"  Win Rate (Last {LOG_FREQ}): {win_rate:.2f}")
            print(f"  Current Epsilon: {current_epsilon:.3f}")
            print(f"  Total Steps Done: {agent.steps_done}")
            print(f"  Buffer Size: {len(agent.replay_buffer)}")
            print(f"  Elapsed Time: {elapsed_time:.1f}s")
            print(f"-------------------\n")
            wins = 0 # Reset win count for the next logging interval
    
    torch.save(agent.policy_net.state_dict(), "/output/dqn_pokemon_policy.pth")
    torch.save(agent.target_net.state_dict(), "/output/dqn_pokemon_target.pth")

    env.close()
    print("\n--- Training Finished ---")
    print(f"  Total reward accumulated: {total_reward}")
    print(f"  Total steps taken: {agent.steps_done}")
    elapsed_time = time.time() - start_time
    print(f"  Total training time: {elapsed_time:.1f}s")

    
if __name__ == "__main__":
    print("--- Script Execution Started ---")
    print("Reminder: Make sure a Pokemon Showdown server is running locally!")
    print(f"Script will train for N_BATTLES = {N_BATTLES}")
    print(f"Hyperparameters: BATCH_SIZE={BATCH_SIZE}, GAMMA={GAMMA}, EPS_START={EPSILON_START}, \
          EPS_END={EPSILON_END}, EPS_DECAY={EPSILON_DECAY}, TAU={TAU}, LR={LR}, \
          BUFFER_SIZE={BUFFER_SIZE}, TARGET_UPDATE_FREQ={TARGET_UPDATE_FREQ}")
    print("-" * 30)

    # Run the asynchronous training function
    try:
        # Use asyncio.run() in Python 3.7+ which handles loop creation/closing
        asyncio.run(train_agent(N_BATTLES))
    except KeyboardInterrupt:
        print("\n--- Training Interrupted By User ---")
    except Exception as e:
        print(f"\n--- An Error Occurred During Training ---")
        print(f"Error Type: {type(e).__name__}")
        print(f"Error Details: {e}")
        import traceback
        traceback.print_exc() # Print full traceback for debugging
    finally:
        print("\n--- Script Execution Finished ---")
        # Optional: Add code here to save the trained model
        # try:
        #     if agent: # Check if agent exists
        #         torch.save(agent.policy_net.state_dict(), "dqn_pokemon_policy.pth")
        #         torch.save(agent.target_net.state_dict(), "dqn_pokemon_target.pth")
        #         print("Model state dictionaries saved to .pth files.")
        # except NameError:
        #     print("Agent variable not defined, cannot save model.")
        # except Exception as e:
        #     print(f"Error saving model: {e}")