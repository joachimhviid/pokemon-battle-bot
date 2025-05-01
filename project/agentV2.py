import numpy as np
import math, time, random, inspect, os, asyncio, traceback

from collections import deque, namedtuple
from gymnasium import spaces
from typing import List, Optional, Union, Dict

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for plotting
import matplotlib.pyplot as plt


from poke_env import Player, AccountConfiguration, ServerConfiguration
from poke_env.environment import AbstractBattle, Status, Move, Pokemon
from poke_env.player import Gen9EnvSinglePlayer, RandomPlayer, MaxBasePowerPlayer
from poke_env.environment.pokemon_type import PokemonType
from poke_env.player.battle_order import BattleOrder, DefaultBattleOrder


# --- 1. Hyperparameters ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

N_BATTLES = 1000000   # Total number of battles to train for (Reduced for faster testing)
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
DEBUG_STEPS = 0        # Print step details for the first N steps of each episode
BATTLE_TIMEOUT = 300.0
LOAD_MODEL = True

# State and Action Space Sizes (Matching runtime observations)
NUM_TYPES = 18 # Standard number of Pokemon types
# (2*HP + 2*TeamSize + 4*Types + 2*Status + Tera) * 2 sides = (2+2+4*18+2*7+2)*2 = (4+72+14+2)*2 = 92*2 = 184?
STATE_SIZE = 184  # 92 if doing single battles, 184 if doing double battles
# print(f"Using STATE_SIZE: {STATE_SIZE}")
ACTION_SPACE_SIZE = 30 
# print(f"Using ACTION_SPACE_SIZE: {ACTION_SPACE_SIZE}")

# --- Server/Account Config ---
BATTLE_FORMAT = "gen9vgc2025regg" # vgc2025regg
SERVER_CONF = ServerConfiguration("ws://localhost:8000/showdown/websocket", None)# type: ignore # Assuming default local server
# Ensure unique names for concurrent runs if needed
OPP_ACC_CONF = AccountConfiguration(f"FixedTeamOpp-VGC", None)
AGENT_ACC_CONF = AccountConfiguration(f"VGC-DQNAgent-{random.randint(0,10000)}", None) # Make agent name unique too

WOLFEY_TEAM = """

    Urshifu-Rapid-Strike @ Focus Sash
    Level: 50
    Ability: unseen-fist
    Tera Type: Water    
    EVs: 4 Hp / 252 Atk / 252 Spe
    IVs: 31 Hp / 31 Atk / 31 Def / 0 SpA / 31 SpD / 31 Spe
    Adamant Nature
    - Detect
    - Close Combat
    - Aqua Jet
    - Surging Strikes

    Calyrex-Shadow @ Covert Cloak
    Level: 50
    Ability: As One (spectrier)
    Tera Type: Ghost
    EVs: 140 Hp / 4 Def / 100 SpA / 12 SpD / 252 Spe
    IVs: 31 Hp / 0 Atk / 31 Def / 31 SpA / 31 SpD / 31 Spe
    Timid Nature
    - Nasty Plot
    - Psyshock
    - Astral Barrage
    - Protect

    Incineroar @ Safety Goggles
    Level: 50
    Ability: Intimidate
    Tera Type: Ghost
    EVs: 252 Hp / 180 Def / 76 SpD
    IVs: 31 Hp / 31 Atk / 31 Def / 31 SpA / 31 SpD / 0 Spe
    Sassy Nature
    - Fake Out
    - Parting Shot
    - Knock Off
    - Helping Hand

    Rillaboom @ Assault Vest
    Level: 50
    Ability: Grassy Surge
    Tera Type: Fire
    EVs: 244 Hp / 36 Atk / 4 Def / 220 SpD / 4 Spe
    IVs: 31 Hp / 31 Atk / 31 Def / 0 SpA / 31 SpD / 31 Spe
    Adamant Nature
    - Fake Out
    - U-turn
    - Wood Hammer
    - Grassy Glide

    Farigiraf @ Throat Spray
    Level: 50
    Ability: Armor Tail
    Tera Type: Psychic
    EVs: 228 Hp / 12 Def / 156 SpA / 112 SpD
    IVs: 31 Hp / 0 Atk / 31 Def / 31 SpA / 31 SpD / 31 Spe
    Modest Nature
    - Protect
    - Hyper Voice
    - Trick Room
    - Dazzling Gleam

    Raging Bolt @ Booster Energy
    Level: 50
    Ability: Protosynthesis
    Tera Type: Fairy
    EVs: 196 Hp / 108 Def / 196 SpA / 4 SpD / 4 Spe
    IVs: 31 Hp / 20 Atk / 31 Def / 31 SpA / 31 SpD / 31 Spe
    Modest Nature
    - Thunderclap
    - Thunderbolt
    - Protect
    - Dragon Pulse
"""

def print_args(func_name, *args, **kwargs):
    pass
    # print(f"--- Entering {func_name} ---")
    # if args:
    #     print(f"  Args: {args}")
    # if kwargs:
    #     try:
    #         class_name = func_name.split('.')[0]
    #         cls = globals().get(class_name)
    #         if cls and hasattr(cls, '__init__'):
    #              sig = inspect.signature(cls.__init__)
    #              bound_args = sig.bind_partial(*args, **kwargs)
    #              bound_args.apply_defaults()
    #              print("  Keyword Args (Bound):")
    #              for name, value in bound_args.arguments.items():
    #                  if name == 'self': continue
    #                  value_repr = repr(value)
    #                  if len(value_repr) > 100: value_repr = value_repr[:97] + "..."
    #                  print(f"    {name}: {value_repr}")
    #         else:
    #              raise ValueError("Class or __init__ not found for signature inspection")
    #     except Exception as e:
    #         print(f"  Keyword Args: {kwargs}")
    # print(f"--- Finished Args for {func_name} ---")

class DQN(nn.Module):
    """Deep Q-Network model."""
    def __init__(self, n_observations, n_actions):
        #print_args("DQN.__init__", n_observations=n_observations, n_actions=n_actions)
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 512)
        self.layer2 = nn.Linear(512, 256)
        self.layer3 = nn.Linear(256, n_actions)
        #print(f"  DQN Initialized: Input Size={n_observations}, Output Size={n_actions}")


    def forward(self, x):
        if x.dtype != torch.float32:
             x = x.to(torch.float32)
        if x.shape[1] != self.layer1.in_features:
             raise RuntimeError(f"Shape mismatch in DQN forward! Input shape {x.shape} does not match layer1 expected input features {self.layer1.in_features}.")
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)


Experience = namedtuple('Experience', ('state', 'action', 'reward', 'next_state', 'done'))

class ReplayBuffer:
    def __init__(self, capacity):
        # print_args("ReplayBuffer.__init__", capacity=capacity) # Keep prints minimal for now
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        state, action, reward, next_state, done = args
        if isinstance(state, torch.Tensor): 
            state = state.cpu().numpy()
        if next_state is not None and isinstance(next_state, torch.Tensor): 
            next_state = next_state.cpu().numpy()

        if state is not None and (len(state.shape) != 1 or state.shape[0] != STATE_SIZE):
            #print(f"ERROR: Pushing state with wrong size {state.shape} to buffer! Expected {STATE_SIZE}.")
            return
        if next_state is not None and (len(next_state.shape) != 1 or next_state.shape[0] != STATE_SIZE):
            #print(f"ERROR: Pushing next_state with wrong size {next_state.shape} to buffer! Expected {STATE_SIZE}.")
            return

        self.memory.append(Experience(state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQNAgent:
    """Manages the DQN model, replay buffer, and learning process."""
    def __init__(self, state_dim, action_dim):
        # print_args("DQNAgent.__init__", state_dim=state_dim, action_dim=action_dim) # Keep prints minimal
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.policy_net = DQN(state_dim, action_dim).to(DEVICE)
        self.target_net = DQN(state_dim, action_dim).to(DEVICE)
        
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LR, amsgrad=True)

        self.replay_buffer = ReplayBuffer(BUFFER_SIZE)
        self.steps_done = 0

    def select_action_index(self, state):
        sample = random.random()
        epsilon_treshold = EPSILON_END + (EPSILON_START - EPSILON_END) * \
            math.exp(-1. * self.steps_done / EPSILON_DECAY)
        
        self.steps_done += 1

        if sample > epsilon_treshold: # Exploit
            with torch.no_grad():
                state_tensor = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
                q_values = self.policy_net(state_tensor)
                action_index = q_values.max(1)[1].view(1, 1)
                return action_index.item()
        else: # Explore
            return random.randrange(self.action_dim)
        
    def learn(self):
        if len(self.replay_buffer) < BATCH_SIZE:
            return None

        experiences = self.replay_buffer.sample(BATCH_SIZE)
        batch = Experience(*zip(*experiences))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)),
                                      device=DEVICE, dtype=torch.bool)
        non_final_next_states_list = [s for s in batch.next_state if s is not None]

        if non_final_next_states_list:
            # Stack numpy arrays first, then convert to tensor
            np_states = np.array(non_final_next_states_list)
            if len(np_states.shape) != 2 or np_states.shape[1] != STATE_SIZE:
                #print(f"ERROR: Non-final next states have wrong dimension {np_states.shape[1]} in learn! Expected {self.state_dim}")
                return None
            non_final_next_states = torch.tensor(np_states, dtype=torch.float32, device=DEVICE)
        else:
            non_final_next_states = torch.empty((0, self.state_dim), dtype=torch.float32, device=DEVICE)

        # Stack numpy arrays first, then convert to tensor
        state_batch_np = np.array(batch.state)

        if len(state_batch_np.shape) != 2 or state_batch_np.shape[1] != STATE_SIZE:
            #print(f"ERROR: State batch has wrong shape {state_batch_np.shape} in learn! Expected ({BATCH_SIZE}, {self.state_dim})")
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
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key] * (1-TAU)
        self.target_net.load_state_dict(target_net_state_dict)

class MyAgent(Player):
    """Custom agent environment wrapping Gen9EnvSinglePlayer."""
    def __init__(self, dqn_agent_logic: DQNAgent, 
                log_lists: Dict[str, list],
                *args, **kwargs):
        
        super().__init__(*args, **kwargs)
        print("  MyAgent initialized.")

        self.type_map = {t.name.lower(): i for i, t in enumerate(PokemonType) if i < NUM_TYPES}
        if len(self.type_map) != NUM_TYPES:
            print(f"Warning: Expected {NUM_TYPES} types in PokemonType enum, but found {len(self.type_map)}.")

        # Correct status map keys to match poke-env standard status names (uppercase)
        self.status_map = {status.name: i for i, status in enumerate(Status)}

        # RL Components
        self.dqn_agent = dqn_agent_logic
        self.log_lists = log_lists
        
        # State Tracking
        self._current_battle: Optional[AbstractBattle] = None
        self._last_battle_state: Optional[AbstractBattle] = None
        self._current_episode_reward: float = 0.0
        self._last_action_index: Optional[int] = None
        self._battles_completed: int = 0
        self._battle_finished_event = asyncio.Event()

    def embed_battle_doubles(self, battle: AbstractBattle) -> np.ndarray:

        features = []

        my_team_size = len([p for p in battle.team.values() if p.fainted is False]) / 6.0
        opp_team_size = len([p for p in battle.opponent_team.values() if p.fainted is False]) / 6.0
        features.extend([my_team_size, opp_team_size]) # 2 Features

        default_pkmn_features = np.zeros(1 + (NUM_TYPES * 2) + len(self.status_map)) # Hp + Types + Status
        for i in range(2): # Loop through the two active slots 
            for player_pkmn, opp_pkmn in [(battle.active_pokemon, battle.opponent_active_pokemon)]:
                pkmn = player_pkmn[i] if i < len(player_pkmn) else None
                if pkmn:
                    hp = pkmn.current_hp_fraction
                    types = np.zeros(NUM_TYPES * 2)
                    if pkmn.type_1:
                        idx = self.type_map.get(pkmn.type_1.name.lower())
                        if idx is not None:
                            types[idx] = 1.0
                    if pkmn.type_2:
                        idx = self.type_map.get(pkmn.type_2.name.lower())
                        if idx is not None:
                            types[idx + NUM_TYPES] = 1.0
                    my_status = np.zeros(len(self.status_map))
                    if pkmn.status:
                        status_idx = self.status_map.get(pkmn.status.name, 0)
                        if status_idx is not None:
                            my_status[status_idx] = 1.0
                    features.extend([hp])
                    features.extend(types)
                    features.extend(my_status)
                else:
                    features.extend(default_pkmn_features)
            
        my_tera_avail = 1.0 if battle.can_tera else 0.0
        opp_tera_avail = 1.0 if battle._opponent_can_terrastallize else 0.0
        features.extend([my_tera_avail, opp_tera_avail]) # 2 Features

        state = np.array(features, dtype=np.float32)


        global STATE_SIZE

        if state.shape[0] != STATE_SIZE:
            print(f"\n!!! CRITICAL WARNING: embed_battle_doubles produced state size {state.shape[0]}, but STATE_SIZE_DOUBLES is {STATE_SIZE}. !!!\n")
            
        if state.shape[0] < STATE_SIZE:
            #print(f"  Padding state to match STATE_SIZE_DOUBLES.")
            state = np.pad(state, (0, STATE_SIZE - state.shape[0]), 'constant')
        elif state.shape[0] > STATE_SIZE:
            #print(f"  Truncating state to match STATE_SIZE_DOUBLES.")
            state = state[:STATE_SIZE]

        return state
    
    def calc_reward(self, last_battle: AbstractBattle | None, current_battle: AbstractBattle) -> float:
        """
        Calculates a reward signal based on changes between battle states.
        Includes:
        - Large reward/penalty for winning/losing the battle.
        - Smaller rewards/penalties for fainting opponent/own Pokemon.
        - Smaller rewards/penalties for HP changes.
        """
        if current_battle is None:
            return 0.0 # Battle ended or error
        
        final_reward = 0.0
        if current_battle.finished:
            if current_battle.won:
                final_reward = 10.0
            elif current_battle.lost:
                final_reward = -15.0

        # These are calculated based on the change from the last state
        intermediate_reward = 0.0
        if last_battle is not None: # Need previous state for comparison

            current_my_fainted = sum(1 for p in current_battle.team.values() if p.fainted)
            last_my_fainted = sum(1 for p in last_battle.team.values() if p.fainted)
            intermediate_reward -= (current_my_fainted - last_my_fainted) * 1.0 # Penalty = 1.0 per own faint

            current_opp_fainted = sum(1 for p in current_battle.opponent_team.values() if p.fainted)
            last_opp_fainted = sum(1 for p in last_battle.opponent_team.values() if p.fainted)
            intermediate_reward += (current_opp_fainted - last_opp_fainted) * 1.0 # Reward = 1.0 per opponent faint

            for i in range(2):
                my_active_now = current_battle.active_pokemon[i] if i < len(current_battle.active_pokemon) else None
                my_active_before = last_battle.active_pokemon[i] if i < len(last_battle.active_pokemon) else None
                opp_active_now = current_battle.opponent_active_pokemon[i] if i < len(current_battle.opponent_active_pokemon) else None
                opp_active_before = last_battle.opponent_active_pokemon[i] if i < len(last_battle.opponent_active_pokemon) else None

                if (opp_active_now and opp_active_before and 
                    hasattr(opp_active_now, "species") and hasattr(opp_active_before, "species") and
                    opp_active_now.species == opp_active_before.species and
                    hasattr(opp_active_now, "current_hp_fraction") and 
                    hasattr(opp_active_before, "current_hp_fraction")):
                    hp_drop_opp = opp_active_before.current_hp_fraction - opp_active_now.current_hp_fraction
                    intermediate_reward += max(0, hp_drop_opp) * 0.5 # Reward = 0.5 * (% HP damage dealt)

                if (my_active_now and my_active_before and
                    hasattr(my_active_now, "species") and hasattr(my_active_before, "species") and
                    my_active_now.species == my_active_before.species and
                    hasattr(my_active_now, "current_hp_fraction") and
                    hasattr(my_active_before, "current_hp_fraction")):
                    hp_drop_me = my_active_before.current_hp_fraction - my_active_now.current_hp_fraction
                    intermediate_reward -= max(0, hp_drop_me) * 0.5   


        # --- Total Reward for the Step ---
        total_step_reward = intermediate_reward + final_reward
        # total_step_reward = np.clip(total_step_reward, -15.0, 15.0)
        return total_step_reward
    
    def choose_move(self, battle: AbstractBattle) -> BattleOrder:
        """Choose a move based on the current battle state."""
        self._current_battle = battle
        step_reward = 0.0
        state = None
        state_embedding_for_buffer = None
        if self._last_battle_state is not None:
            step_reward = self.calc_reward(self._last_battle_state, battle)
            self._current_episode_reward += step_reward

            state = self.embed_battle_doubles(self._last_battle_state)
            next_state = self.embed_battle_doubles(battle)
            action_idx = self._last_action_index if self._last_action_index is not None else -1

            if state is not None and action_idx != -1:
                self.dqn_agent.replay_buffer.push(state, action_idx, step_reward, next_state, battle.finished)

            loss = self.dqn_agent.learn()
            if loss is not None and "recent_losses_log" in self.log_lists: 
                self.log_lists["recent_losses_log"].append(loss)

        if battle.turn == 0:
            # print(f"DEBUG: Turn 0 - Handling Team Preview (Default Order)")
            self._last_battle_state = battle
            self._current_episode_reward = 0.0
            return DefaultBattleOrder()

        if not battle.available_moves and not battle.available_switches and not battle.trapped:
            # print(f"Warning: No moves or switches available, and not trapped?.")
            self._last_battle_state = battle
            return DefaultBattleOrder()

        current_state_embedding = self.embed_battle_doubles(battle)

        chosen_action_idx = self.dqn_agent.select_action_index(current_state_embedding)
        self._last_action_index = chosen_action_idx

        final_order = self.choose_random_move(battle)
        self._last_battle_state = battle

        return final_order
    
    def _battle_finished_callback(self, battle: AbstractBattle) -> None:
        """Called when a battle ends."""
        #print(f"--- Battle Finished: {battle.battle_tag} ---")
        self._battles_completed += 1

        # --- 1. Final Reward Calculation & Experience ---
        final_reward = 0.0
        if self._last_battle_state: # If there was a previous state
            final_reward = self.calc_reward(self._last_battle_state, battle)
            self._current_episode_reward += final_reward

            # Store final experience
            last_state_embedding = self.embed_battle_doubles(self._last_battle_state)
            last_action = self._last_action_index if self._last_action_index is not None else -1
            if last_action != -1:
                 self.dqn_agent.replay_buffer.push(last_state_embedding, last_action, final_reward, None, True) # next_state is None, done is True
        else:
            print("Warning: Battle finished but _last_battle_state was None.")

        battle_won = battle.won
        result = "Won" if battle_won else "Lost" if battle.lost else "Draw/Other"
        #print(f"  Result: {result}, Total Reward: {self._current_episode_reward:.4f}")

        # Access logging lists passed during __init__
        self.log_lists['recent_rewards'].append(self._current_episode_reward)
        self.log_lists['recent_wins'].append(1 if battle_won else 0)

        # if battle_won:
        #     self.log_lists["total_wins"] = self.log_lists.get("total_wins", 0 ) + 1
        
        # --- 3. Periodic Updates & Plotting ---
        if self._battles_completed % TARGET_UPDATE_FREQ == 0:
            #print(f"Updating target network (Battle {self._battles_completed})...")
            self.dqn_agent.update_target_network()

        if self._battles_completed % LOG_FREQ == 0:
            #print(f"Logging progress (Battle {self._battles_completed})...")
            avg_reward = sum(self.log_lists['recent_rewards']) / len(self.log_lists['recent_rewards']) if self.log_lists['recent_rewards'] else 0
            avg_win_rate = sum(self.log_lists['recent_wins']) / len(self.log_lists['recent_wins']) if self.log_lists['recent_wins'] else 0

            self.log_lists['episode_log_points'].append(self._battles_completed)
            self.log_lists['avg_rewards_log'].append(avg_reward)
            
            recent_losses = self.log_lists.get("recent_losses", deque(maxlen=LOG_FREQ))
            avg_loss = sum(recent_losses) / len(recent_losses) if recent_losses else 0
            self.log_lists['avg_losses_log'].append(avg_loss) # Add placeholder/rolling avg
            self.log_lists['win_rates_log'].append(avg_win_rate)

            # Generate periodic plot (needs plot function and lists)
            plot_training_results(
                self.log_lists['episode_log_points'],
                self.log_lists['avg_rewards_log'],
                self.log_lists['avg_losses_log'],
                self.log_lists['win_rates_log'],
                save_path=self.log_lists['periodic_plot_path'] # Get path from log_lists dict
            )
        
        #Signal that the battle is finished
        self._battle_finished_event.set()

        # --- Reset State ---
        self._last_battle_state = None
        self._current_battle = None
        self._current_episode_reward = 0.0
        self._last_action_index = None

    def reset_battle_event(self):
        self._battle_finished_event.clear()

async def train_agent(n_battles_to_run):
    #print(f"\n--- Starting Training Function: train_agent ---")
    #print(f"  n_battles_to_run: {n_battles_to_run}")
    start_time = time.time()

    dqn_logic = DQNAgent(STATE_SIZE, ACTION_SPACE_SIZE)
    #print(f"--- DQNAgent Initialized ---")    

    save_dir = "./project/output/models"
    plot_dir = "./project/output" 
    final_plot_path = os.path.join(plot_dir, "FP_Doubles.png") 
    policy_path = os.path.join(save_dir, "Model_Policy_Doubles-v1.0.0.pth")
    target_path = os.path.join(save_dir, "Model_target_Doubles-v1.0.0.pth")
    os.makedirs(save_dir, exist_ok=True) # Ensure directory exists
    os.makedirs(plot_dir, exist_ok=True) # Ensure directory exists

    if LOAD_MODEL and os.path.exists(policy_path) and os.path.exists(target_path):
        try:
            #print(f"Loading existing models from: {save_dir}")
            dqn_logic.policy_net.load_state_dict(torch.load(policy_path, map_location=DEVICE))
            dqn_logic.target_net.load_state_dict(torch.load(target_path, map_location=DEVICE))
            dqn_logic.policy_net.eval() # Set policy net to eval mode if loaded (or train mode if continuing training)
            dqn_logic.target_net.eval() # Target net is always in eval mode
            #print("VGC Models loaded successfully.")
        except Exception as e:
            pass
            #print(f"Error loading models: {e}. Starting training from scratch.")
    else:
        if LOAD_MODEL:
            #print("No existing models found or LOAD_MODEL is False. Starting training from scratch.")
            pass
        else:
            #print("LOAD_MODEL is False. Starting training from scratch.")
            pass

    opponent = RandomPlayer(
        account_configuration=OPP_ACC_CONF,
        battle_format=BATTLE_FORMAT,
        server_configuration=SERVER_CONF,
        team=WOLFEY_TEAM
    )

    log_lists = {
        'episode_log_points': [],
        'avg_rewards_log': [],
        'avg_losses_log': [], # Loss avg needs refinement
        'win_rates_log': [],
        'recent_rewards': deque(maxlen=LOG_FREQ),
        'recent_wins': deque(maxlen=LOG_FREQ),
        'recent_losses': deque(maxlen=LOG_FREQ * 100), # Store loss deque here? Or in dqn_logic?
        'final_plot_path': final_plot_path
    }

    agent = MyAgent(
        dqn_agent_logic=dqn_logic,
        log_lists=log_lists,
        account_configuration=AGENT_ACC_CONF,
        battle_format=BATTLE_FORMAT,
        log_level=15,
        server_configuration=SERVER_CONF,
        team=WOLFEY_TEAM
    )

    #print(f"--- Starting {n_battles_to_run} Battles ---")
    opponent_task = None
    try:
        print("Starting opponent task to accept challenges...")
        async def opponent_accept_loop(player, agent_username):
            while True:
                try:
                    await player.accept_challenges(agent_username, n_battles_to_run)
                except asyncio.CancelledError:
                    #print("Opponent accept loop cancelled.")
                    break # Exit loop if cancelled
                except Exception as e:
                    #print(f"Error in opponent accept loop: {e}")
                    await asyncio.sleep(5) # Wait before retrying

        opponent_task = asyncio.create_task(
             opponent_accept_loop(opponent, agent.username)
        )
        await asyncio.sleep(3)
        #print("\n--- Starting Manual Battle Loop ---")
        
        agent.reset_battle_event()

        await agent.send_challenges(opponent.username, n_battles_to_run)

        try:
            await asyncio.wait_for(agent._battle_finished_event.wait(), timeout=BATTLE_TIMEOUT)
        except asyncio.TimeoutError:
            #print(f"Battle {n_battles_to_run + 1} timed out after {BATTLE_TIMEOUT} seconds.")
            pass
        except Exception as e:
            #print(f"Error during battle: {e}")
            traceback.print_exc()
        
        await asyncio.sleep(1) # Small delay to avoid overwhelming the serverter

    except asyncio.CancelledError:
        #print("Training loop cancelled.")
        traceback.print_exc()
    except Exception as e:
        #print(f"Error during battle loop: {e}")
        traceback.print_exc()
    finally:
        #print("\n--- Cleaning up tasks ---")
        if opponent_task and not opponent_task.done():
            opponent_task.cancel()
            try:
                await opponent_task # Allow cancellation to propagate
            except asyncio.CancelledError:
                #print("Opponent task successfully cancelled.")
                traceback.print_exc()
            except Exception as e:
                #print(f"Error during opponent task cancellation: {e}")
                traceback.print_exc()
        #print("\n--- Finished Battle Tasks ---")
    
    #print(f"--- Finished {n_battles_to_run} Battles ---")
    
    total_wins = sum(log_lists['recent_wins'])
    total_battles_completed = agent._battles_completed
    final_win_rate = sum(1 for w in log_lists['win_rates_log'] if w > 0.5) / len(log_lists["win_rates_log"]) if log_lists["win_rates_log"] else 0.0
    #print(f"Total Battles Completed: {total_battles_completed}")
    #print(f"Total Wins: {total_wins}")
    #print(f"Overall Win Rate: {final_win_rate:.2%}")
    
    try:
        torch.save(dqn_logic.policy_net.state_dict(), policy_path)
        torch.save(dqn_logic.target_net.state_dict(), target_path)
        #print(f"Models saved successfully to {save_dir}.")
    except Exception as e:
        #print(f"Error saving models: {e}.")
        traceback.print_exc()
    
    plot_training_results(
        log_lists['episode_log_points'],
        log_lists['avg_rewards_log'],
        log_lists['avg_losses_log'],
        log_lists['win_rates_log'],
        save_path=final_plot_path
    )

    print(f"--- Training Complete ---")

def plot_training_results(episodes, rewards, losses, win_rates, save_path):
    """Generates and saves plots for training metrics."""
    #print(f"Generating plot with {len(episodes)} data points...")
    fig, axs = plt.subplots(3, 1, figsize=(12, 18), sharex=True)

    # Plot Average Reward
    axs[0].plot(episodes, rewards, label=f'Avg Reward per {LOG_FREQ} Episodes', color='blue', marker='o', linestyle='-')
    axs[0].set_ylabel('Average Reward')
    axs[0].set_title('Training Progress')
    axs[0].grid(True, linestyle='--', alpha=0.6)
    axs[0].legend()

    # Plot Average Loss
    axs[1].plot(episodes, losses, label=f'Avg Loss (Rolling)', color='red', marker='x', linestyle='-')
    axs[1].set_ylabel('Average Loss')
    axs[1].grid(True, linestyle='--', alpha=0.6)
    axs[1].legend()

    # Plot Win Rate
    axs[2].plot(episodes, win_rates, label=f'Win Rate per {LOG_FREQ} Episodes', color='green', marker='s', linestyle='-')
    axs[2].set_ylabel('Win Rate')
    axs[2].set_xlabel(f'Episodes (Intervals of {LOG_FREQ})')
    axs[2].grid(True, linestyle='--', alpha=0.6)
    axs[2].legend()
    axs[2].set_ylim(0, 1) # Win rate is between 0 and 1

    plt.tight_layout(rect=(0, 0.03, 1, 0.95)) # Adjust layout to fit title
    try:
        plt.savefig(save_path)
        #print(f"Training plot saved to {save_path}")
    except Exception as e:
        #print(f"Error saving plot: {e}")
        traceback.print_exc()
    plt.close(fig) # Close the figure to free memory

if __name__ == "__main__":
    #print("--- Script Execution Started ---")
    #print(f"Attempting to train for {N_BATTLES} VGC battles ({BATTLE_FORMAT})")
    #print(f"Using DOUBLES STATE_SIZE: {STATE_SIZE}") # Use correct variable name
    #print(f"Load existing model: {LOAD_MODEL}")

    try:
        # Run the doubles training function
        asyncio.run(train_agent(N_BATTLES))
    except KeyboardInterrupt: 
        #print("\n--- Training Interrupted By User ---")
        traceback.print_exc()
    except ConnectionRefusedError: 
        #print("\n--- CONNECTION ERROR ---\nCould not connect to the Pokemon Showdown server.\nPlease ensure the server is running at the configured address.")
        traceback.print_exc()
    except Exception as e: 
        #print(f"\n--- An Unhandled Error Occurred During Execution ---\nError Type: {type(e).__name__}\nError Details: {e}"); traceback.print_exc()
        traceback.print_exc()
    finally: 
        print("\n--- Script Execution Finished ---")

