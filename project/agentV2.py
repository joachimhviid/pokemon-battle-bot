import numpy as np
import math, time, random, inspect, os, asyncio
from collections import deque, namedtuple

from gymnasium import spaces

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from poke_env.environment import observation
from poke_env import Player, AccountConfiguration, ServerConfiguration
from poke_env.environment import AbstractBattle
from poke_env.player import Gen9EnvSinglePlayer, RandomPlayer, MaxBasePowerPlayer
from poke_env.environment.pokemon_type import PokemonType


# --- 1. Hyperparameters ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

N_BATTLES = 10         # Total number of battles to train for (Reduced for faster testing)
BATCH_SIZE = 128       # Number of experiences to sample from buffer for learning
GAMMA = 0.99           # Discount factor for future rewards
EPSILON_START = 0.9    # Starting exploration rate (Changed name for consistency)
EPSILON_END = 0.05     # Minimum exploration rate (Changed name for consistency)
EPSILON_DECAY = 10000  # How fast the exploration rate decays (higher = slower decay)
TAU = 0.005            # Soft update factor for target network
LR = 1e-4              # Learning rate for the optimizer
BUFFER_SIZE = 20000    # Max size of the replay buffer
TARGET_UPDATE_FREQ = 5 # How often (in episodes/battles) to update target network weights
LOG_FREQ = 10          # How often (in episodes/battles) to print progress (Increased freq for testing)
DEBUG_STEPS = 5        # Print step details for the first N steps of each episode

# State and Action Space Sizes (Matching runtime observations)
NUM_TYPES = 18 # Standard number of Pokemon types
STATE_SIZE = 90 # Set based on runtime error for Gen9RandomBattle
print(f"Using STATE_SIZE: {STATE_SIZE}")
ACTION_SPACE_SIZE = 30 # Increased for Gen9RandomBattle
print(f"Using ACTION_SPACE_SIZE: {ACTION_SPACE_SIZE}")

# --- Server/Account Config ---
BATTLE_FORMAT = "gen9randombattle" # Battle format for the environment
SERVER_CONF = ServerConfiguration("ws://localhost:8000/showdown/websocket", None)# type: ignore # Assuming default local server
# Ensure unique names for concurrent runs if needed
OPP_ACC_CONF = AccountConfiguration(f"RandomOpponent", None)
AGENT_ACC_CONF = AccountConfiguration(f"DQNAgent-{random.randint(0,10000)}", None) # Make agent name unique too

# --- Helper Function to Print Args/Kwargs ---
# (Keep existing print_args function - omitted here for brevity)
def print_args(func_name, *args, **kwargs):
    print(f"--- Entering {func_name} ---")
    if args:
        print(f"  Args: {args}")
    if kwargs:
        try:
            class_name = func_name.split('.')[0]
            cls = globals().get(class_name)
            if cls and hasattr(cls, '__init__'):
                 sig = inspect.signature(cls.__init__)
                 bound_args = sig.bind_partial(*args, **kwargs)
                 bound_args.apply_defaults()
                 print("  Keyword Args (Bound):")
                 for name, value in bound_args.arguments.items():
                     if name == 'self': continue
                     value_repr = repr(value)
                     if len(value_repr) > 100: value_repr = value_repr[:97] + "..."
                     print(f"    {name}: {value_repr}")
            else:
                 raise ValueError("Class or __init__ not found for signature inspection")
        except Exception as e:
            print(f"  Keyword Args: {kwargs}")
    print(f"--- Finished Args for {func_name} ---")


# --- 4. DQN Model --- (Defined before Agent)
class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        print_args("DQN.__init__", n_observations=n_observations, n_actions=n_actions)
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 256)
        self.layer2 = nn.Linear(256, 128)
        self.layer3 = nn.Linear(128, n_actions)
        print(f"  DQN Initialized: Input Size={n_observations}, Output Size={n_actions}")
        print(f"  DQN Layers: Linear({n_observations}, 256) -> ReLU -> Linear(256, 128) -> ReLU -> Linear(128, {n_actions})")

    def forward(self, x):
        if x.dtype != torch.float32:
             x = x.to(torch.float32)
        if x.shape[1] != self.layer1.in_features:
             raise RuntimeError(f"Shape mismatch in DQN forward! Input shape {x.shape} does not match layer1 expected input features {self.layer1.in_features}.")
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

# --- 2. Environment Wrapper ---
# Use correct base class for Gen 9
class MyAgent(Gen9EnvSinglePlayer):
    def __init__(self, *args, **kwargs):
        print_args("MyAgent.__init__", *args, **kwargs) # Use actual class name
        super().__init__(*args, **kwargs)
        print("  MyAgent initialized.")

        self.type_map = {t.name.lower(): i for i, t in enumerate(PokemonType) if i < NUM_TYPES}
        if len(self.type_map) != NUM_TYPES:
             print(f"Warning: Expected {NUM_TYPES} types in PokemonType enum, but found {len(self.type_map)}.")

        # Correct status map keys to match poke-env standard status names (uppercase)
        self.status_map = {None: 0, 'PAR': 1, 'BRN': 2, 'FRZ': 3, 'PSN': 4, 'SLP': 5, 'TOX': 6}


    def embed_battle(self, battle: AbstractBattle) -> np.ndarray:
        """
        Computes a numerical representation of the battle state.
        Output vector size must match STATE_SIZE.
        """
        my_active = battle.active_pokemon
        opp_active = battle.opponent_active_pokemon
        my_hp = my_active.current_hp_fraction if my_active else 0.0
        opp_hp = opp_active.current_hp_fraction if opp_active else 0.0
        my_team_size = len([p for p in battle.team.values() if p.fainted is False]) / 6.0
        opp_team_size = len([p for p in battle.opponent_team.values() if p.fainted is False]) / 6.0

        my_types_vector = np.zeros(NUM_TYPES * 2)
        if my_active:
            if my_active.type_1:
                type_1_name = my_active.type_1.name.lower()
                if type_1_name in self.type_map: my_types_vector[self.type_map[type_1_name]] = 1
            if my_active.type_2:
                type_2_name = my_active.type_2.name.lower()
                if type_2_name in self.type_map: my_types_vector[self.type_map[type_2_name] + NUM_TYPES] = 1

        opp_types_vector = np.zeros(NUM_TYPES * 2)
        if opp_active:
            if opp_active.type_1:
                type_1_name = opp_active.type_1.name.lower()
                if type_1_name in self.type_map: opp_types_vector[self.type_map[type_1_name]] = 1
            if opp_active.type_2:
                type_2_name = opp_active.type_2.name.lower()
                if type_2_name in self.type_map: opp_types_vector[self.type_map[type_2_name] + NUM_TYPES] = 1

        # Use corrected status_map
        my_status_vector = np.zeros(len(self.status_map))
        opp_status_vector = np.zeros(len(self.status_map))
        # Get status name (which is uppercase in poke-env Status enum)
        my_status_name = my_active.status.name if my_active and my_active.status else None
        opp_status_name = opp_active.status.name if opp_active and opp_active.status else None
        my_status_vector[self.status_map.get(my_status_name, 0)] = 1
        opp_status_vector[self.status_map.get(opp_status_name, 0)] = 1

        # Prepare components for concatenation
        c1 = np.array([my_hp, opp_hp], dtype=np.float32)
        c2 = np.array([my_team_size, opp_team_size], dtype=np.float32)
        c3 = my_types_vector.astype(np.float32)
        c4 = opp_types_vector.astype(np.float32)
        c5 = my_status_vector.astype(np.float32)
        c6 = opp_status_vector.astype(np.float32)

        # --- Debug Print for component shapes (Uncommented) ---
        print(f"DEBUG embed_battle shapes: c1={c1.shape}, c2={c2.shape}, c3={c3.shape}, c4={c4.shape}, c5={c5.shape}, c6={c6.shape}")
        # Expected: (2,), (2,), (36,), (36,), (7,), (7,) -> Total 88

        state = np.concatenate([c1, c2, c3, c4, c5, c6])

        if state.shape[0] != STATE_SIZE:
             print(f"\n!!! CRITICAL WARNING: embed_battle produced state size {state.shape[0]}, but STATE_SIZE is set to {STATE_SIZE}. !!!")

        # Pad or truncate to ensure state size is exactly STATE_SIZE (90)
        if state.shape[0] < STATE_SIZE:
            # print(f"Warning: Padding state from {state.shape[0]} to {STATE_SIZE}") # Optional print
            padding = np.zeros(STATE_SIZE - state.shape[0], dtype=np.float32)
            state = np.concatenate([state, padding])
        elif state.shape[0] > STATE_SIZE:
            # print(f"Warning: Truncating state from {state.shape[0]} to {STATE_SIZE}") # Optional print
            state = state[:STATE_SIZE]

        return state.astype(np.float32)

    def describe_embedding(self) -> spaces.Space:
        """
        Returns the description of the embedding space (observation space).
        Uses the global STATE_SIZE.
        """
        low = np.zeros(STATE_SIZE, dtype=np.float32)
        high = np.ones(STATE_SIZE, dtype=np.float32)
        print(f"--- Describing Embedding Space ---")
        print(f"  Shape: ({STATE_SIZE},)") # Should reflect 90
        print(f"  Data Type: np.float32")
        print(f"  Low Bound: 0.0 (for all elements)")
        print(f"  High Bound: 1.0 (for all elements)")
        print(f"---------------------------------")
        return spaces.Box(low=low, high=high, shape=(STATE_SIZE,), dtype=np.float32)

    def calc_reward(self, last_battle: AbstractBattle, current_battle: AbstractBattle) -> float:
        """
        Calculates the reward difference between the last state and the current state.
        """
        def get_reward_for_battle(battle: AbstractBattle) -> float:
            if battle is None: return 0.0
            if battle.won: return 10.0
            elif battle.lost: return -10.0
            else: return 0.0

        current_reward = get_reward_for_battle(current_battle)
        last_reward = get_reward_for_battle(last_battle)
        step_reward = current_reward - last_reward
        return step_reward


    def action_to_move(self, action_idx: int, battle: AbstractBattle):
        """Converts action index back to poke-env move/switch order."""
        current_moves = battle.available_moves
        current_switches = battle.available_switches
        n_moves = len(current_moves)
        n_switches = len(current_switches)
        total_actions = n_moves + n_switches
        max_legal_idx = total_actions - 1 if total_actions > 0 else -1 # Handle case with no actions

        if 0 <= action_idx < n_moves:
            return self.create_order(current_moves[action_idx])
        elif n_moves <= action_idx < total_actions:
            switch_idx = action_idx - n_moves
            return self.create_order(current_switches[switch_idx])
        else:
            # --- Add detailed debugging ---
            print(f"--- DEBUG action_to_move ELSE block ---")
            print(f"  action_idx: {action_idx}")
            print(f"  n_moves: {n_moves}")
            print(f"  n_switches: {n_switches}")
            print(f"  total_actions: {total_actions}")
            print(f"  Condition failed: action_idx ({action_idx}) < 0 or >= total_actions ({total_actions})")
            print(f"--- END DEBUG ---")
            # Original warning and fallback
            print(f"Warning: Action index {action_idx} selected is out of bounds "
                  f"(Moves: {n_moves}, Switches: {n_switches}, Max Valid Idx: {max_legal_idx}). "
                  f"Choosing random valid action.")
            return self.choose_random_move(battle) # Fallback

# --- 3. Replay Buffer ---
Experience = namedtuple('Experience', ('state', 'action', 'reward', 'next_state', 'done'))

class ReplayBuffer:
    def __init__(self, capacity):
        # print_args("ReplayBuffer.__init__", capacity=capacity) # Keep prints minimal for now
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        state, action, reward, next_state, done = args
        if isinstance(state, torch.Tensor): state = state.cpu().numpy()
        if next_state is not None and isinstance(next_state, torch.Tensor): next_state = next_state.cpu().numpy()

        if state is not None and state.shape[0] != STATE_SIZE:
             print(f"ERROR: Pushing state with wrong size {state.shape} to buffer! Expected {STATE_SIZE}.")
             return
        if next_state is not None and next_state.shape[0] != STATE_SIZE:
             print(f"ERROR: Pushing next_state with wrong size {next_state.shape} to buffer! Expected {STATE_SIZE}.")
             return

        self.memory.append(Experience(state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


# --- 5. DQN Agent ---
class DQNAgent:
    def __init__(self, state_dim, action_dim):
        # print_args("DQNAgent.__init__", state_dim=state_dim, action_dim=action_dim) # Keep prints minimal
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.policy_net = DQN(state_dim, action_dim).to(DEVICE)
        # print("DQNAgent: policy network created.")
        self.target_net = DQN(state_dim, action_dim).to(DEVICE)
        # print("DQNAgent: target network created.")
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        # print("DQNAgent: target network weights loaded and set to eval mode.")

        # Use AdamW as before, or Adam as in user code
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LR, amsgrad=True) # Matching user code
        # print(f"DQNAgent: Optimizer Adam created with lr={LR}, amsgrad=True.")
        self.replay_buffer = ReplayBuffer(BUFFER_SIZE)
        # print(f"DQNAgent: Replay buffer created with capacity={BUFFER_SIZE}.")
        self.steps_done = 0
        # print("DQNAgent: initialization complete.")

    def select_action(self, state, available_action_indices):
        sample = random.random()
        epsilon_threshold = EPSILON_END + (EPSILON_START - EPSILON_END) * \
            math.exp(-1.0 * self.steps_done / EPSILON_DECAY)

        self.steps_done += 1

        if not available_action_indices:
             print("Warning: No available actions in select_action. Returning default action 0.")
             return torch.tensor([[0]], device=DEVICE, dtype=torch.long)

        if sample > epsilon_threshold: # Exploit
            with torch.no_grad():
                state_tensor = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)

                # print(f"DEBUG: state_tensor shape before policy_net call: {state_tensor.shape}")
                # expected_features = self.policy_net.layer1.in_features
                # print(f"DEBUG: policy_net expects input features: {expected_features}")

                q_values = self.policy_net(state_tensor)
                mask = torch.full_like(q_values, -float("inf"))
                valid_indices_tensor = torch.tensor(available_action_indices, device=DEVICE, dtype=torch.long)

                if valid_indices_tensor.numel() > 0:
                     valid_indices_tensor = valid_indices_tensor[valid_indices_tensor < q_values.shape[1]] # Ensure index is within network output size
                     if valid_indices_tensor.numel() > 0:
                         mask[0, valid_indices_tensor] = q_values[0, valid_indices_tensor]

                if torch.isinf(mask).all():
                    print("Warning: Masking resulted in all -inf Q-values during exploitation. Choosing random valid action.")
                    action_idx = random.choice(available_action_indices) # Choose from valid list
                    return torch.tensor([[action_idx]], device=DEVICE, dtype=torch.long)
                else:
                    best_action_idx = mask.argmax(dim=1)
                    return best_action_idx.view(1, 1)
        else: # Explore
            action_idx = random.choice(available_action_indices) # Choose from valid list
            # Correct variable name from user code: action_index -> action_idx
            return torch.tensor([[action_idx]], device=DEVICE, dtype=torch.long)

    def learn(self):
        if len(self.replay_buffer) < BATCH_SIZE:
            return None

        experiences = self.replay_buffer.sample(BATCH_SIZE)
        batch = Experience(*zip(*experiences))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)),
                                      device=DEVICE, dtype=torch.bool)

        # Corrected handling from user code: Convert list of numpy arrays directly
        non_final_next_states_list = [s for s in batch.next_state if s is not None]

        if non_final_next_states_list:
            # Stack numpy arrays first, then convert to tensor
            np_states = np.array(non_final_next_states_list)
            if np_states.shape[1] != self.state_dim:
                 print(f"ERROR: Non-final next states have wrong dimension {np_states.shape[1]} in learn! Expected {self.state_dim}")
                 return None
            non_final_next_states = torch.tensor(np_states, dtype=torch.float32, device=DEVICE)
        else:
            non_final_next_states = torch.empty((0, self.state_dim), dtype=torch.float32, device=DEVICE)

        # Stack numpy arrays first, then convert to tensor
        state_batch_np = np.array(batch.state)
        if len(state_batch_np.shape) < 2 or state_batch_np.shape[1] != self.state_dim:
             print(f"ERROR: State batch has wrong shape {state_batch_np.shape} in learn! Expected ({BATCH_SIZE}, {self.state_dim})")
             return None
        state_batch = torch.tensor(state_batch_np, dtype=torch.float32, device=DEVICE)


        action_batch = torch.tensor(batch.action, dtype=torch.long, device=DEVICE).unsqueeze(1)
        reward_batch = torch.tensor(batch.reward, dtype=torch.float32, device=DEVICE).unsqueeze(1)

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
        # Correct typo: policcy -> policy
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key] * (1-TAU)
        self.target_net.load_state_dict(target_net_state_dict)


async def train_agent(n_battles_to_run):
    print(f"\n--- Starting Training Function: train_agent ---")
    print(f"  n_battles_to_run: {n_battles_to_run}")
    start_time = time.time()

    opponent = RandomPlayer(
        account_configuration=OPP_ACC_CONF,
        battle_format=BATTLE_FORMAT,
        server_configuration=SERVER_CONF
    )
    print(f"  Opponent Player ({type(opponent).__name__}) Config:")
    print(f"    Battle Format: {BATTLE_FORMAT}")

    env_player = MyAgent(
        opponent=opponent,
        account_configuration=AGENT_ACC_CONF,
        battle_format=BATTLE_FORMAT,
        log_level=20, # Keep log level if desired
        server_configuration=SERVER_CONF,
        start_challenging=True

    )
    print(f"  RL Player ({type(env_player).__name__}) set up.")

    _ = env_player.describe_embedding()
    print(f"  Environment embedding space described.")

    env = env_player
    print(f"  Environment set to RL Player instance.")

    print(f"\n--- Initializing DQNAgent ---")
    agent = DQNAgent(STATE_SIZE, ACTION_SPACE_SIZE)
    print(f"--- DQNAgent Initialized ---\n")

    print(f"--- Starting Training Loop for {n_battles_to_run} Battles ---")
    wins = 0
    total_reward = 0
    recent_rewards = deque(maxlen=LOG_FREQ)
    # recent_losses = deque(maxlen=LOG_FREQ) # Uncomment if tracking loss
    # total_losses = 0.0 # Uncomment if tracking loss

    for episode in range(1, n_battles_to_run + 1):
        state_dict, info = env.reset()

        if env.current_battle is None:
            print(f"ERROR: env.current_battle is None after reset in Episode {episode}! Skipping episode.")
            await asyncio.sleep(1) # Add a small delay before potentially retrying
            continue # Skip to the next episode

        state = env.embed_battle(env.current_battle)
        if episode == 1:
            print(f"  Initial state shape from embed_battle: {state.shape}")
        if state.shape[0] != STATE_SIZE:
             print(f"  ERROR: Initial state shape {state.shape} does not match STATE_SIZE {STATE_SIZE}! Stopping.")
             break

        done = False
        episode_reward = 0.0
        episode_loss = 0.0
        steps = 0 # Use 'steps' for steps where learning occurs
        turn = 0

        while not done:
            turn += 1
            assert isinstance(env.action_space, spaces.Discrete), \
                f"Expected Discrete action space, but got {type(env.action_space)}"
            current_action_space_size = env.action_space.n
            # Correct typo: available_action_indencies -> available_action_indices
            available_action_indices = list(range(current_action_space_size))

            if state is None:
                 print(f"ERROR: State is None before selecting action in Ep {episode}, Turn {turn}. Breaking episode.")
                 break
            if isinstance(state, torch.Tensor): state = state.cpu().numpy()
            if state.shape[0] != STATE_SIZE:
                 print(f"ERROR: State shape is {state.shape} before select_action! Expected ({STATE_SIZE},). Breaking episode.")
                 break

            action_tensor = agent.select_action(state, available_action_indices)
            action = action_tensor.item()


            if action >= current_action_space_size:
                 if turn <= DEBUG_STEPS:
                     print(f"  [Ep {episode}, Turn {turn}] Warning: Agent chose invalid action {action} (>= {current_action_space_size}). Choosing random.")
                 # Ensure random choice is made from non-empty list
                 if not available_action_indices:
                      print("ERROR: No available actions to choose randomly from!")
                      break
                 action = random.choice(available_action_indices)
                 # No need to update action_tensor, 'action' (int) is used

            next_state_dict, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            next_state = env.embed_battle(env.current_battle) if not done else None

            if turn <= DEBUG_STEPS:
                print(f"  [Ep {episode}, Turn {turn}] Action Chosen: {action}, Reward: {reward:.4f}, Done: {done}" # Use Action Chosen
                      f", Available Actions: {current_action_space_size}")
                if next_state is not None:
                     print(f"     Next State (Embed shape): {next_state.shape}")
                else:
                     print("     Next State: None (Episode Ended)")

            agent.replay_buffer.push(state, action, reward, next_state, done)
            state = next_state

            loss = agent.learn()
            if loss is not None:
                episode_loss += loss
                steps += 1 # Increment steps only when learning occurs
            elif len(agent.replay_buffer) >= BATCH_SIZE:
                 print(f"  [Ep {episode}, Turn {turn}] Learning step skipped/failed (Buffer full but learn returned None).")


            episode_reward += reward

            if done:
                break

        total_reward += episode_reward
        recent_rewards.append(episode_reward)
        # Safely check win condition
        battle_won = hasattr(env.current_battle, 'won') and env.current_battle.won
        if battle_won: wins += 1

        if episode % TARGET_UPDATE_FREQ == 0:
            agent.update_target_network()

        if episode % LOG_FREQ == 0:
            avg_loss = (episode_loss / steps) if steps > 0 else 0 # Use steps where learning happened
            avg_reward = sum(recent_rewards) / len(recent_rewards) if recent_rewards else 0
            win_rate = wins / LOG_FREQ # This tracks wins over the last LOG_FREQ episodes
            # Correct typo: elasped_time -> elapsed_time
            elapsed_time = time.time() - start_time
            # Correct typo: current_epsilon -> current_eps_threshold (or similar)
            current_epsilon_val = EPSILON_END + (EPSILON_START - EPSILON_END) * \
                math.exp(-1.0 * agent.steps_done / EPSILON_DECAY)
            print(f"\n--- Episode Log ---")
            print(f"  Episode: {episode}/{n_battles_to_run}")
            print(f"  Avg Reward (Last {LOG_FREQ}): {avg_reward:.3f}")
            print(f"  Avg Loss (This Episode): {avg_loss:.5f}")
            print(f"  Win Rate (Last {LOG_FREQ}): {win_rate:.2f}")
            print(f"  Current Epsilon: {current_epsilon_val:.3f}") # Use correct variable
            print(f"  Total Steps Done (Agent Decisions): {agent.steps_done}") # Clarify steps_done
            print(f"  Buffer Size: {len(agent.replay_buffer)}")
            print(f"  Elapsed Time: {elapsed_time:.1f}s")
            print(f"-------------------\n")
            wins = 0 # Reset win count for the next logging interval

    # --- Save Model ---
    # Use relative paths
    save_dir = "./project/output" # Save in current directory
    os.makedirs(save_dir, exist_ok=True) # Ensure directory exists
    policy_path = os.path.join(save_dir, "Model-V0.1.pth")
    target_path = os.path.join(save_dir, "Model-V0.1.pth")
    torch.save(agent.policy_net.state_dict(), policy_path)
    torch.save(agent.target_net.state_dict(), target_path)
    print(f"Models saved to {policy_path} and {target_path}")


    env.close()
    print("\n--- Training Finished ---")
    print(f"  Total reward accumulated: {total_reward}")
    print(f"  Total steps taken (Agent Decisions): {agent.steps_done}") # Clarify steps_done
    elapsed_time = time.time() - start_time
    print(f"  Total training time: {elapsed_time:.1f}s")


# --- 7. Main Execution ---
if __name__ == "__main__":
    print("--- Script Execution Started ---")
    print("Reminder: Make sure a Pokemon Showdown server is running locally!")
    print(f"Script will train for N_BATTLES = {N_BATTLES}")
    print(f"Using STATE_SIZE = {STATE_SIZE}, ACTION_SPACE_SIZE = {ACTION_SPACE_SIZE}")
    print(f"Hyperparameters: BATCH_SIZE={BATCH_SIZE}, GAMMA={GAMMA}, EPS_START={EPSILON_START}, \
EPS_END={EPSILON_END}, EPS_DECAY={EPSILON_DECAY}, TAU={TAU}, LR={LR}, \
BUFFER_SIZE={BUFFER_SIZE}, TARGET_UPDATE_FREQ={TARGET_UPDATE_FREQ}")
    print("-" * 30)

    try:
        asyncio.run(train_agent(N_BATTLES))
    except KeyboardInterrupt:
        print("\n--- Training Interrupted By User ---")
    except Exception as e:
        print(f"\n--- An Error Occurred During Training ---")
        print(f"Error Type: {type(e).__name__}")
        print(f"Error Details: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\n--- Script Execution Finished ---")
        # Saving moved to end of train_agent function

