import numpy as np
import torch 
import torch.nn as nn 
import torch.optim as optim  
import random
import asyncio
import json
import time
import os

from gymnasium import spaces
from collections import deque
from matplotlib import pyplot as plt
from dqn import DQN
from poke_env import Player, AccountConfiguration, ServerConfiguration
from poke_env.environment import Battle
from poke_env.environment.abstract_battle import AbstractBattle
from poke_env.player import MaxBasePowerPlayer, SimpleHeuristicsPlayer, RandomPlayer
from poke_env.environment.move_category import MoveCategory
from poke_env.player.openai_api import OpenAIGymEnv


class PokemonAgent(Player):
    def __init__(self, state_size, action_size, battle_format="doublesubers", server_configuration=None, account_configuration=None, team=None):
        super(PokemonAgent, self).__init__(battle_format=battle_format, 
                                           server_configuration=server_configuration, 
                                           account_configuration=account_configuration,
                                           team=team)
        
        # Define action/state space
        self.state_size = state_size
        self.action_size = action_size # Number of possible moves/switches agent can choose

        # Initialize the model and memory
        self.model = DQN(self.state_size, self.action_size)
        self.target_model = DQN(self.state_size, self.action_size)
        self.target_model.load_state_dict(self.model.state_dict()) # Initialize target model weights
        self.target_model.eval() # Target model is only for inference

        self.gamma = 0.95
        self.epsilon= 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.05
        self.learning_rate = 0.001
        self.tau = 0.005 # For soft update of target model
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.MSELoss()

        self.memory = deque(maxlen=100000)
        self.batch_size = 64

        self.training_errors = []
        self.current_battle_state_action = None
      
    def embed_battle(self, battle: Battle) -> np.ndarray:
        active = battle.active_pokemon
        opponent = battle.opponent_active_pokemon

        active_hp = active.current_hp_fraction if active else 0.0
        opponent_hp = opponent.current_hp_fraction if opponent else 0.0
    
        state = np.array([active_hp, opponent_hp], dtype=np.float32)
        if self.state_size > 16:
             # Pad with zeros if state_size is larger (temporary fix)
             padding = np.zeros(self.state_size - 2, dtype=np.float32)
             state = np.concatenate((state, padding))
        elif self.state_size < 2:
             state = state[:self.state_size] # Truncate if smaller

        return state
    
    def choose_move(self, battle: Battle):

        state = self.embed_battle(battle)
        state_tensor = torch.FloatTensor(state).unsqueeze(1) # Add batch dimension

        possible_actions = battle.available_moves + battle.available_switches
        action_index = 0
        choose_action = None
        self.current_battle_state_action = None

        if battle.force_switch:
            valid_switches = [sw for sw in battle.available_switches if sw.current_hp_fraction > 0]
            if not valid_switches:
                choose_action_order = self.choose_random_move(battle)
                return choose_action_order
            choose_action = random.choice(valid_switches)
            try: action_index = possible_actions.index(choose_action)
            except ValueError: action_index = -1
            
            self.current_battle_state_action = (state, action_index)
            return self.create_order(choose_action)

        can_dmg = any(
            move.base_power > 0 and move.category != MoveCategory.STATUS 
            for move in battle.available_moves
        )

        if not can_dmg and battle.available_switches:
            best_switch = max(
                battle.available_switches,
                key=lambda p: p.current_hp_fraction and p.damage     
            ) # A Simple Heuristic
            for i, action in enumerate(possible_actions):
                if action == best_switch:
                    return self.create_order(action), i

            return self.choose_random_move(battle), -1

        if not possible_actions:
            print(f"Warning: No possible actions available. Choosing random move.")
            choose_action_order = self.choose_random_move(battle)
            self.current_battle_state_action = (state, -1)
            return choose_action_order

        if random.random() < self.epsilon:
            #Exploration
            action_index = random.randrange(len(possible_actions))
        else:
            #Exploitation
            self.model.eval()
            with torch.no_grad():
                q_values = self.model(state_tensor)
            self.model.train() # Set back to training mode

            q_values = q_values.squeeze(0)
            valid_action_mask = torch.full_like(q_values, -float('inf'))
            valid_action_mask[:len(possible_actions)] = 0
            masked_q_values = q_values + valid_action_mask
            action_index = torch.argmax(masked_q_values).item()
        
        
        if action_index >= len(possible_actions) or action_index < 0:
            action_index = 0 # Fallback to first move if move index is invalid
        
        choose_action = possible_actions[action_index]
        self.current_battle_state_action = (state, action_index)

        return self.create_order(choose_action)
    
    def _store_transition(self, state, action_index, reward, next_state, done):
        if isinstance(state, torch.tensor):  state_np = state.cpu().numpy()
        if isinstance(next_state, torch.tensor): next_state_np = next_state.cpu().numpy()

        self.memory.append((state_np, action_index, reward, next_state_np, done))

    def learn_from_memory(self):
        if len(self.memory) < self.batch_size:
            return None

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, done = zip(*batch)
        
        states_tensor = torch.FloatTensor(np.array(states))
        actions_tensor = torch.LongTensor(actions).unsqueeze(1)
        rewards_tensor = torch.FloatTensor(rewards).unsqueeze(1)
        next_states_tensor = torch.FloatTensor(np.array(next_states))
        done_tensor = torch.FloatTensor(done).unsqueeze(1)

        current_q_values = self.model(states_tensor).gather(1, actions_tensor)

        with torch.no_grad():
            next_q_values = self.target_model(next_states_tensor).max(1)[0].unsqueeze(1)

        target_q_values = rewards_tensor + self.gamma * next_q_values * (~done_tensor)

        loss = self.loss_fn(current_q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()

        self.optimizer.step()

        for target_param, local_param in zip(self.target_model.parameters(), self.model.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)

        self.training_errors.append(loss.item())
        return loss.item()

    def _calc_reward(self, battle: AbstractBattle, previous_hp_ratio: float) -> float:
        current_hp_ratio = battle.active_pokemon.current_hp_fraction if battle.active_pokemon else 0.0
        reward = (current_hp_ratio - previous_hp_ratio)

        if battle.won:
            return reward + 10
        elif battle.lost:
            return reward - 10
        else:
            return reward   

    def save_model(self, path="project/output/", name="Model-V1-0-0.pth"):
        os.makedirs(path, exist_ok=True)
        save_path = os.path.join(path, name)
        torch.save(self.model.state_dict(), save_path)
        print(f"Model saved to {path}")

    def load_model(self, path="project/output/Model-V1-0-0.pth"):
        if os.path.exists(path):
            self.model.load_state_dict(torch.load(path))
            self.target_model.load_state_dict(self.model.state_dict())
            self.model.eval()
            self.target_model.eval()
            print(f"Model loaded from {path}")
        else:
            print(f"Model file not found at {path}")

def plot_training(num_total_episodes, rewards, losses, win_rates, save_path="project/output/training_plot.png"):

    if not rewards and not losses and not win_rates:
        print("No data to plot.")
        return

    num_episodes_axis = range(1, len(rewards) + 1)
    fig, axs = plt.subplots(3, 1, figsize=(10, 15), sharex=True)

    if rewards:
        axs[0].plot(num_episodes_axis, rewards, label="Total Reward per Episode", color="Green")
    axs[0].set_title("Episode Rewards")
    axs[0].set_ylabel("Total Reward")
    axs[0].legend()
    axs[0].grid(True)


    loss_episodes = [i + 1 for i, loss in enumerate(losses) if loss is not None]
    valid_losses = [loss for loss in losses if loss is not None]
    if valid_losses:
        axs[1].plot(loss_episodes, valid_losses, label="Average Loss", color="Red", marker=".", linestyle="None")
        axs[1].set_ylabel("Average Loss")
        axs[1].set_title("Training Loss")
        axs[1].legend()
        axs[1].grid(True)
    else:
        axs[1].set_title("Training Loss (No Data)")

    if win_rates:
        axs[2].plot(num_episodes_axis, win_rates, label="Overall Win Rate", color="blue")
    axs[2].set_xlabel(f"Episode up to {len(rewards)}")
    axs[2].set_ylabel("Win rate")
    axs[2].set_title("Agent win rate")
    axs[2].legend()
    axs[2].grid(True)
    axs[2].set_ylim(0, 1)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    print(f"Plots saved to {save_path}")
    plt.close(fig)

# --- Corrected Custom Training Loop (from agent_py_fix_v6) ---
async def custom_training_loop(agent: PokemonAgent, opponent: MaxBasePowerPlayer, episodes: int = 1000):
    all_win_rates = []; 
    all_avg_losses = []; 
    all_episode_rewards = []
    start_time = time.time(); 
    train_step_counter = 0
    print(f"Starting training for {episodes} episodes..."); 
    print(f"Agent: {agent.username}, Opponent: {opponent.username}"); 
    print(f"State size: {agent.state_size}, Action size: {agent.action_size}")

    for episode in range(episodes):
        episode_start_time = time.time()
        print(f"\n--- Starting Episode {episode + 1}/{episodes} ---")
        episode_reward_sum = 0.0; 
        current_episode_losses = []
        battle = Battle(battle_tag="Empty", username=agent.username, logger=None, save_replays=False, gen=9); 
        state = None; 
        action_index = -1

        try:
            # --- Initiate ONE battle using challenge/accept ---
            print("Setting up battle: Opponent challenges, Agent accepts...")
            # --- Ensure correct method: challenge_player ---
            asyncio.run(await opponent.send_challenges(dqn_agent.username , n_challenges=10))
            asyncio.run(await dqn_agent.accept_challenges(opponent.username , n_challenges=10))

            # Start opponent's battle task

            #CHALLENGE_TIMEOUT = 45.0 # Timeout in seconds
            #print(f"Waiting for challenge/accept (timeout: {CHALLENGE_TIMEOUT}s)...")
            """
            done, pending = await asyncio.wait(
                [challenge_task_obj, accept_task_obj],
                return_when=asyncio.FIRST_COMPLETED,
                timeout=CHALLENGE_TIMEOUT
            )

            # --- Logging and result checking ---
            accepted_battles = None; 
            challenge_error = None

            print(f" asyncio.wait finished. Done tasks: {done}, Pending tasks: {pending}") # Debug print
            for task in done:
                try:
                    result = task.result()
                    if task == accept_task_obj:
                        if isinstance(result, list) and result: accepted_battles = result; print(" Challenge accepted by agent.")
                        elif result is None: print(" Agent accept task finished but no battle returned (None).")
                        else: print(f" Unexpected result from accept_task: {result}")
                    elif task == challenge_task_obj: print(" Challenge task finished.")
                except Exception as e:
                    task_name = "Accept Task" if task == accept_task_obj else "Challenge Task"
                    print(f" Error captured from {task_name}: {e}"); challenge_error = e
            if pending:
                print(f" Cancelling pending tasks: {pending}")
                for task in pending:
                    task.cancel()
                    try: await task
                    except asyncio.CancelledError: pass
                    except Exception as e: print(f" Error during cancellation of pending task: {e}")

            if not accepted_battles:
                print("Error: Battle could not be established.")
                if challenge_error: print(f" Underlying error detail: {challenge_error}")
                if not done: print(" Timeout likely occurred.")
                continue # Skip rest of try block

            # --- Battle Started ---
            battle = accepted_battles[0]; 
            print(f"Battle accepted: {battle.battle_tag}")
"""
            # --- Turn-based Loop (Keep as is) ---
            turn = 0
            while battle.finished == False:
                turn += 1
                last_state_action = agent.current_battle_state_action
                valid_last_action = last_state_action is not None and last_state_action[1] != -1
                _ = agent.choose_move(battle)
                # asyncio.sleep(0.05) # ADJUST AS NEEDED
                next_state = agent.embed_battle(battle)
                done = battle
                reward = 0.0
                if last_state_action:
                    state_before_action, _ = last_state_action
                    hp_before_action = state_before_action[0]
                    hp_after_action = battle.active_pokemon.current_hp_fraction if battle.active_pokemon else 0.0
                    reward = (hp_after_action - hp_before_action)
                    if battle.won : reward += 1.0
                    elif battle.lost: reward -= 1.0
                episode_reward_sum += reward
                if valid_last_action:
                    state, action_index = last_state_action
                    if 0 <= action_index < agent.action_size: agent._store_transition(state, action_index, reward, next_state, done)
                loss = agent.learn_from_memory()
                if loss is not None: current_episode_losses.append(loss); train_step_counter += 1
                if done: break

        #except asyncio.CancelledError: 
        #    print("Training cancelled."); break
        except Exception as e:
            current_turn = turn if 'turn' in locals() else 'N/A'
            print(f"Error during episode {episode + 1}, turn {current_turn}: {e}")
            import traceback; 
            traceback.print_exc()

        # --- Crucial Finally Block for Cleanup ---

        # --- Log Episode Results (Keep as is) ---
        episode_won = battle.won if battle and hasattr(battle, 'won') else False
        current_win_rate = agent.n_won_battles / agent.n_finished_battles if agent.n_finished_battles > 0 else 0
        all_win_rates.append(current_win_rate)
        avg_episode_loss = np.mean(current_episode_losses) if current_episode_losses else None
        all_avg_losses.append(avg_episode_loss)
        all_episode_rewards.append(episode_reward_sum)
        episode_duration = time.time() - episode_start_time
        result_str = 'Win' if episode_won else 'Loss' if battle else 'Error/Incomplete'
        print(f"Episode {episode + 1} finished. Result: {result_str}. Reward: {episode_reward_sum:.2f}. Win rate: {current_win_rate:.2f}. Epsilon: {agent.epsilon:.4f}. Duration: {episode_duration:.2f}s")
        if avg_episode_loss is not None: print(f"Average Loss this episode: {avg_episode_loss:.4f}")
        agent.epsilon = max(agent.epsilon_min, agent.epsilon * agent.epsilon_decay)

    # --- After all episodes (Keep as is) ---
    end_time = time.time(); print(f"\nTraining finished after {end_time - start_time:.2f} seconds.")
    print(f"Final Win Rate: {all_win_rates[-1]:.2f}" if all_win_rates else "N/A")
    plot_training(len(all_episode_rewards), all_episode_rewards, all_avg_losses, all_win_rates)
    agent.save_model()

def convert_json_to_showdown(team_data: list) -> str:
        showdown_team = ""
        stat_map = {
            "hp": "HP",
            "atk": "Atk",
            "attack": "Atk",
            "def": "Def",
            "defense": "Def",
            "spa": "SpA",
            "sp_attack": "SpA",
            "special-attack": "SpA",
            "spd": "SpD",
            "sp_defense": "SpD",
            "special-defense": "SpD",
            "spe": "Spe",
            "speed": "Spe"
        }

        for pokemon in team_data:
            name = pokemon["name"].title()
            item = pokemon.get("held_item", "None").replace("_", " ").title()
            ability = pokemon["ability"]["name"].replace("-", " ").title()
            level = pokemon.get("level", 100)
            nature = pokemon["nature"].capitalize()
            evs = pokemon["evs"]
            ivs = pokemon.get("ivs", {})

            # Header
            showdown_team += f"{name} @ {item}\n"
            showdown_team += f"Ability: {ability}\n"
            showdown_team += f"Level: {level}\n"

            # EVs
            ev_parts = []
            for stat, val in evs.items():
                if val > 0:
                    stat_name = stat_map.get(stat.lower(), stat.upper())
                    ev_parts.append(f"{val} {stat_name}")
            showdown_team += f"EVs: {' / '.join(ev_parts)}\n"

            # Nature
            showdown_team += f"{nature} Nature\n"

            # Moves
            for move in pokemon["moves"]:
                showdown_team += f"- {move['name'].replace('-', ' ').title()}\n"
            
            showdown_team += "\n"

        return showdown_team.strip()


if __name__ == "__main__":
    # To start the server REMEBMER to use: NODE pokemon-showdown start --no-security
    print("Ensure the local Pokemon Showdown server is running.")
    print("Command: node pokemon-showdown start --no-security")

    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    output_dir = "project/output"
    os.makedirs(output_dir, exist_ok=True)

    BATTLE_FORMAT = "vgc2025regg"
    NUM_EPISODES = 100
    
    server_config = ServerConfiguration("ws://localhost:8000/showdown/websocket", None)
    opponent_config = AccountConfiguration(f"Random_{random.randint(0,10000)}", None)
    agent_config = AccountConfiguration(f"Agent_{random.randint(0,10000)}", None)


    my_team = None
    if BATTLE_FORMAT != "gen9randombattle":
        team_path = "project/input/my_team.json"
        try:
            with open(team_path, "r") as f:
                team_data = json.load(f)
            my_team = convert_json_to_showdown(team_data)
            print(f"Loaded team for format: {BATTLE_FORMAT}")
        except FileNotFoundError:
            print(f"Error: Team file not found at {team_path}. Required for format {BATTLE_FORMAT}.")
            exit()
        except Exception as e:
            print(f"Error loading or converting team: {e}")
            exit()

    STATE_SIZE = 2
    ACTION_SIZE = 10

    dqn_agent = PokemonAgent(
        state_size=STATE_SIZE,
        action_size=ACTION_SIZE,
        server_configuration=server_config,
        account_configuration=agent_config,
        team=my_team,
        battle_format=BATTLE_FORMAT
    )

    opponent = MaxBasePowerPlayer(
        server_configuration=server_config,
        battle_format=BATTLE_FORMAT,
        account_configuration=opponent_config,
        team=my_team
    )

    asyncio.run(custom_training_loop(dqn_agent, opponent, episodes=NUM_EPISODES))

    # Optional: Load pre-trained model if available
    # dqn_agent.load_model(path="project/output/Model-V1-0-0.pth")
"""
    print("Starting training loop...")
    try:
        
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
        # Optionally save model even if interrupted
        dqn_agent.save_model()
    except Exception as main_error:
        print(f"\nAn error occurred in the main execution: {main_error}")
        import traceback
        traceback.print_exc()
"""