import numpy as np
import math, time, random, inspect, os, asyncio, traceback
from collections import deque, namedtuple

from gymnasium import spaces

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from poke_env.environment import observation
from poke_env import Player, AccountConfiguration, ServerConfiguration
from poke_env.environment import AbstractBattle, Move
from poke_env.player import Gen9EnvSinglePlayer, RandomPlayer, MaxBasePowerPlayer
from poke_env.environment.pokemon_type import PokemonType
from poke_env.player.battle_order import BattleOrder


# --- 1. Hyperparameters ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

N_BATTLES = 100     # Total number of battles to train for (Reduced for faster testing)
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
STATE_SIZE = 92  # 90 - Set based on runtime error for Gen9RandomBattle
print(f"Using STATE_SIZE: {STATE_SIZE}")
ACTION_SPACE_SIZE = 30 # Increased for Gen9RandomBattle
print(f"Using ACTION_SPACE_SIZE: {ACTION_SPACE_SIZE}")

# --- Server/Account Config ---
BATTLE_FORMAT = "gen9ou" # vgc2025regg
SERVER_CONF = ServerConfiguration("ws://localhost:8000/showdown/websocket", None)# type: ignore # Assuming default local server
# Ensure unique names for concurrent runs if needed
OPP_ACC_CONF = AccountConfiguration(f"FixedTeamOpp-OU", None)
AGENT_ACC_CONF = AccountConfiguration(f"DQNAgent-{random.randint(0,10000)}", None) # Make agent name unique too

Wolfey_TEAM = """

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

AGENT_TEAM = """
Glimmora @ Focus Sash
Ability: Toxic Debris
Tera Type: Ghost
EVs: 252 SpA / 4 SpD / 252 Spe
Timid Nature
- Mortal Spin
- Stealth Rock
- Energy Ball
- Earth Power

Great Tusk @ Booster Energy
Ability: Protosynthesis
Tera Type: Ground
EVs: 252 Atk / 4 SpD / 252 Spe
Jolly Nature
- Headlong Rush
- Close Combat
- Ice Spinner
- Rapid Spin

Iron Valiant @ Booster Energy
Ability: Quark Drive
Tera Type: Fairy
EVs: 252 SpA / 4 SpD / 252 Spe
Timid Nature
- Moonblast
- Psyshock
- Shadow Ball
- Thunderbolt

Kingambit @ Black Glasses
Ability: Supreme Overlord
Tera Type: Dark
EVs: 252 Atk / 4 SpD / 252 Spe
Adamant Nature
- Kowtow Cleave
- Sucker Punch
- Iron Head
- Swords Dance

Dragapult @ Choice Specs
Ability: Infiltrator
Tera Type: Ghost
EVs: 252 SpA / 4 SpD / 252 Spe
Timid Nature
- Shadow Ball
- Draco Meteor
- U-turn
- Thunderbolt

Cinderace @ Heavy-Duty Boots
Ability: Libero
Tera Type: Fire
EVs: 252 Atk / 4 SpD / 252 Spe
Jolly Nature
- Pyro Ball
- U-turn
- Court Change
- Will-O-Wisp
"""

OPPONENT_TEAM = """
Gholdengo @ Choice Scarf
Ability: Good as Gold
Tera Type: Steel
EVs: 252 SpA / 4 SpD / 252 Spe
Timid Nature
- Make It Rain
- Shadow Ball
- Focus Blast
- Trick

Dragonite @ Heavy-Duty Boots
Ability: Multiscale
Tera Type: Normal
EVs: 252 Atk / 4 SpD / 252 Spe
Adamant Nature
- Extreme Speed
- Earthquake
- Dragon Dance
- Ice Spinner

Garganacl @ Leftovers
Ability: Purifying Salt
Tera Type: Fairy
EVs: 252 HP / 4 Def / 252 SpD
Careful Nature
- Salt Cure
- Recover
- Protect
- Earthquake

Iron Moth @ Booster Energy
Ability: Quark Drive
Tera Type: Grass
EVs: 104 Def / 148 SpA / 4 SpD / 252 Spe
Timid Nature
- Fiery Dance
- Sludge Wave
- Energy Ball
- Dazzling Gleam

Ting-Lu @ Leftovers
Ability: Vessel of Ruin
Tera Type: Water
EVs: 252 HP / 4 Def / 252 SpD
Careful Nature
- Stealth Rock
- Spikes
- Ruination
- Earthquake

Meowscarada @ Heavy-Duty Boots
Ability: Protean
Tera Type: Grass
EVs: 252 Atk / 4 SpD / 252 Spe
Jolly Nature
- Flower Trick
- Knock Off
- U-turn
- Spikes
"""

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
        my_tera_avail = 1.0 if battle.can_tera else 0.0
        opp_tera_avail = 1.0 if battle._opponent_can_terrastallize else 0.0

        # Prepare components for concatenation
        c1 = np.array([my_hp, opp_hp], dtype=np.float32)
        c2 = np.array([my_team_size, opp_team_size], dtype=np.float32)
        c3 = my_types_vector.astype(np.float32)
        c4 = opp_types_vector.astype(np.float32)
        c5 = my_status_vector.astype(np.float32)
        c6 = opp_status_vector.astype(np.float32)
        c7 = np.array([my_tera_avail, opp_tera_avail], dtype=np.float32)
        

        # --- Debug Print for component shapes (Uncommented) ---
        print(f"DEBUG embed_battle shapes: c1={c1.shape}, c2={c2.shape}, c3={c3.shape}, c4={c4.shape}, c5={c5.shape}, c6={c6.shape}, c7={c7.shape}")
        # Expected: (2,), (2,), (36,), (36,), (7,), (7,) -> Total 90

        state = np.concatenate([c1, c2, c3, c4, c5, c6, c7]) # Expects 90 ???

        if state.shape[0] != STATE_SIZE:
             print(f"\n!!! CRITICAL WARNING: embed_battle produced state size {state.shape[0]}, but STATE_SIZE is set to {STATE_SIZE}. !!!")
        """
        # Pad or truncate to ensure state size is exactly STATE_SIZE (90)
        if state.shape[0] < STATE_SIZE:
            # print(f"Warning: Padding state from {state.shape[0]} to {STATE_SIZE}") # Optional print
            padding = np.zeros(STATE_SIZE - state.shape[0], dtype=np.float32)
            state = np.concatenate([state, padding])
        elif state.shape[0] > STATE_SIZE:
            # print(f"Warning: Truncating state from {state.shape[0]} to {STATE_SIZE}") # Optional print
            state = state[:STATE_SIZE]
        """
        return state.astype(np.float32)

    def describe_embedding(self) -> spaces.Space:
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
             print(f"ERROR: Pushing state with wrong size {state.shape} to buffer! Expected {STATE_SIZE}.")
             return
        if next_state is not None and (len(next_state.shape) != 1 or next_state.shape[0] != STATE_SIZE):
             print(f"ERROR: Pushing next_state with wrong size {next_state.shape} to buffer! Expected {STATE_SIZE}.")
             return

        self.memory.append(Experience(state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQNAgent:
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
            if len(np_states.shape) != 2 or np_states.shape[1] != self.state_dim:
                 print(f"ERROR: Non-final next states have wrong dimension {np_states.shape[1]} in learn! Expected {self.state_dim}")
                 return None
            non_final_next_states = torch.tensor(np_states, dtype=torch.float32, device=DEVICE)
        else:
            non_final_next_states = torch.empty((0, self.state_dim), dtype=torch.float32, device=DEVICE)

        # Stack numpy arrays first, then convert to tensor
        state_batch_np = np.array(batch.state)
        if len(state_batch_np.shape) != 2 or state_batch_np.shape[1] != self.state_dim:
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
        server_configuration=SERVER_CONF,
        team=OPPONENT_TEAM
    )
    print(f"  Opponent Player ({type(opponent).__name__}) Config:")
    print(f"    Battle Format: {BATTLE_FORMAT}")

    env_player = MyAgent(
        opponent=opponent,
        account_configuration=AGENT_ACC_CONF,
        battle_format=BATTLE_FORMAT,
        log_level=5,     
        server_configuration=SERVER_CONF,
        team=AGENT_TEAM,
        start_challenging=True,
    )
    print(f"  RL Player ({type(env_player).__name__}) set up.")

    print("  Allowing time for players to connect and challenge...")
    await asyncio.sleep(3)

    _ = env_player.describe_embedding()
    env = env_player

    print(f"\n--- Initializing DQNAgent ---")
    agent = DQNAgent(STATE_SIZE, ACTION_SPACE_SIZE)
    print(f"--- DQNAgent Initialized ---\n")

    print(f"--- Starting Training Loop for {n_battles_to_run} Battles ---")
    wins = 0
    total_reward = 0
    recent_rewards = deque(maxlen=LOG_FREQ)


    # --- Lists for plotting ---
    episode_log_points = []
    avg_rewards_log = []
    avg_losses_log = []
    win_rates_log = []
    # --- Interval accumulators ---
    interval_total_loss = 0.0
    interval_total_steps = 0

    for episode in range(1, n_battles_to_run):
        state = None
        done = False
        episode_reward = 0.0
        episode_loss = 0.0
        steps = 0
        turn = 0

        try:
            state_dict, info = env.reset()
            if env.current_battle is None:
                print(f"ERROR: env.current_battle is None after reset in Ep {episode}! Skipping.")
                await asyncio.sleep(1)
            state = env.embed_battle(env.current_battle)
            
            if not isinstance(state, np.ndarray) or state.shape[0] != STATE_SIZE:
                print(f"ERROR: Initial state shape {state.shape} != STATE_SIZE {STATE_SIZE}! Stopping.")
                break
        except Exception as e:
            print(f"ERROR during env.reset()/embed_battle in Ep {episode}: {e}")
            traceback.print_exc(); 
            await asyncio.sleep(5); 
            continue

        while not done:
            turn += 1
            try:
                
                if state is None:
                     print(f"ERROR: State is None at start of turn {turn} Ep {episode}. Breaking.")
                     break
                if isinstance(state, torch.Tensor): 
                    state = state.cpu().numpy()
                if not isinstance(state, np.ndarray) or state.shape[0] != STATE_SIZE:
                     print(f"ERROR: State shape {state.shape} != STATE_SIZE {STATE_SIZE}. Breaking.")
                     break

                assert isinstance(env.action_space, spaces.Discrete), f"Expected Discrete action space, got {type(env.action_space)}"
                current_max_possible_actions = env.action_space.n
                available_action_indices = list(range(current_max_possible_actions))
                action_tensor = agent.select_action(state, available_action_indices)
                action = action_tensor.item() 


                if env.current_battle:
                    current_moves = env.current_battle.available_moves
                    current_switches = env.current_battle.available_switches
                    actual_total_actions = len(current_moves) + len(current_switches)

                    if actual_total_actions == 0 and not env.current_battle.finished:
                        action = 0 # Default to action 0 if nothing else possible
                    elif action >= actual_total_actions:
                        if actual_total_actions > 0:
                            action = random.choice(list(range(actual_total_actions))) # Choose from 0 to actual_total_actions-1
                        else: action = 0

                else:
                    print(f"ERROR: env.current_battle is None before action correction in Ep {episode}, Turn {turn}. Breaking.")
                    break

                next_state_dict, reward, terminated, truncated, info = env.step(action)
                
                done = terminated or truncated
                if not done and env.current_battle is not None:
                    next_state = env.embed_battle(env.current_battle)
                    if not isinstance(next_state, np.ndarray) or next_state.shape[0] != STATE_SIZE:
                        print(f"ERROR: Next state shape {next_state.shape} != STATE_SIZE {STATE_SIZE}. Breaking.")
                        done = True
                        next_state = None
                else:
                    next_state = None

                if state is not None:
                    agent.replay_buffer.push(state, action, reward, next_state, done)
                
                state = next_state
                
                loss = agent.learn()
                if loss is not None:
                    episode_loss += loss
                    steps += 1

                if done: break

            except Exception as e:
                print(f"ERROR during battle loop (Ep {episode}, Turn {turn}): {e}")
                traceback.print_exc(); done = True

        # --- End of Episode ---
        total_reward += episode_reward
        recent_rewards.append(episode_reward)
        battle_won = env.current_battle is not None and hasattr(env.current_battle, 'won') and env.current_battle.won
        if battle_won: 
            wins += 1

        if episode % TARGET_UPDATE_FREQ == 10: 
            agent.update_target_network()

        if episode % LOG_FREQ == 10:
            # Calculate interval metrics
            avg_loss_interval = interval_total_loss / interval_total_steps if interval_total_steps > 0 else 0
            avg_reward_interval = sum(recent_rewards) / len(recent_rewards) if recent_rewards else 0 # recent_rewards already holds last LOG_FREQ
            win_rate_interval = wins / LOG_FREQ # Wins over the last LOG_FREQ episodes
            
            # Append data for plotting
            episode_log_points.append(episode)
            avg_rewards_log.append(avg_reward_interval)
            avg_losses_log.append(avg_loss_interval)
            win_rates_log.append(win_rate_interval)

            elapsed_time = time.time() - start_time
            current_epsilon_val = EPSILON_END + (EPSILON_START - EPSILON_END) * \
                math.exp(-1.0 * agent.steps_done / EPSILON_DECAY)
            print(f"\n--- Episode Log ---")
            print(f"  Episode: {episode}/{n_battles_to_run}")
            print(f"-------------------\n")
            wins = 0
            plot_training_results(episode_log_points, avg_rewards_log, avg_losses_log, win_rates_log)

    plot_training_results(episode_log_points, avg_rewards_log, avg_losses_log, win_rates_log)
    # --- Save Model ---
    save_dir = "./project/output/models"
    os.makedirs(save_dir, exist_ok=True)
    policy_path = os.path.join(save_dir, "Model_Policy-v1.0.0.pth")
    target_path = os.path.join(save_dir, "Model_target-v1.0.0.pth")
    try:
        if 'agent' in locals() and agent is not None:
            torch.save(agent.policy_net.state_dict(), policy_path)
            torch.save(agent.target_net.state_dict(), target_path)
            print(f"Models saved to {policy_path} and {target_path}")
        else: print("Agent not initialized, models not saved.")
    except Exception as e: print(f"Error saving models: {e}")



    env.close()
    print("\n--- Training Finished ---")

def plot_training_results(episodes, rewards, losses, win_rates, save_path="./project/output/training_plot.png"):
    """Generates and saves plots for training metrics."""
    print(f"Generating plot with {len(episodes)} data points...")
    fig, axs = plt.subplots(3, 1, figsize=(10, 15), sharex=True)

    # Plot Average Reward
    axs[0].plot(episodes, rewards, label='Avg Reward per Interval', color='blue')
    axs[0].set_ylabel('Average Reward')
    axs[0].set_title('Training Progress')
    axs[0].grid(True)
    axs[0].legend()

    # Plot Average Loss
    axs[1].plot(episodes, losses, label='Avg Loss per Interval', color='red')
    axs[1].set_ylabel('Average Loss')
    axs[1].grid(True)
    axs[1].legend()

    # Plot Win Rate
    axs[2].plot(episodes, win_rates, label='Win Rate per Interval', color='green')
    axs[2].set_ylabel('Win Rate')
    axs[2].set_xlabel(f'Episodes (Intervals of {LOG_FREQ})')
    axs[2].grid(True)
    axs[2].legend()
    axs[2].set_ylim(0, 1) # Win rate is between 0 and 1

    plt.tight_layout()
    try:
        plt.savefig(save_path)
        print(f"Training plot saved to {save_path}")
    except Exception as e:
        print(f"Error saving plot: {e}")
    plt.close(fig) # Close the figure to free memory


if __name__ == "__main__":
    print("--- Script Execution Started ---")
    print("Reminder: Make sure a Pokemon Showdown server is running locally!")
    print(f"Script will train for N_BATTLES = {N_BATTLES}")
    print(f"Using STATE_SIZE = {STATE_SIZE}, ACTION_SPACE_SIZE = {ACTION_SPACE_SIZE}")

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

