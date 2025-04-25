import json
from datetime import datetime
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm

from project import BattleAgent, BattleEnv, team_1, team_2


def main(episodes: int = 1000, target_update: int = 10, log_name: str = 'test-log.txt'):
    # print(f"CUDA available: {torch.cuda.is_available()}")
    # print(f"Current device: {torch.cuda.current_device()}")
    # print(f"Device count: {torch.cuda.device_count()}")
    # return
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(f"output/{run_id}")
    out_dir.mkdir(parents=True, exist_ok=True)

    env = BattleEnv(player_team=team_1, opponent_team=team_2)

    player_agent = BattleAgent(env)
    opponent_agent = BattleAgent(env)

    env.player_agent = player_agent
    env.opponent_agent = opponent_agent

    # Metrics tracking
    metrics = {
        'episode_rewards': [],
        'player_wins': 0,
        'opponent_wins': 0,
        'episode_lengths': [],
        'player_epsilon': [],
        'opponent_epsilon': [],
        'moving_avg_reward': []
    }

    # Clear the log file at the start of a new run
    log_path = out_dir / log_name
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write("Pokemon Battle Training Log\n\n")

    # Training loop
    for episode in tqdm(range(episodes)):
        state, _ = env.reset()
        done = False
        episode_reward = 0
        steps = 0

        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(f"\nEpisode {episode + 1}\n{'=' * 20}\n")

        while not done:
            steps += 1
            # Player turn
            player_action_mask = env.get_action_mask('player')
            opponent_action_mask = env.get_action_mask('opponent')
            # print(f'Player mask {player_action_mask}')
            # print(f'Opponent mask {opponent_action_mask}')

            action = player_agent.choose_action(state, player_action_mask)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Store the transition in memory
            player_agent.store_transition(state, action, next_state, reward, done)
            opponent_agent.store_transition(state, action, next_state, -reward, done)  # Note the negative reward

            # Train both agents
            if not done:
                player_agent.train(player_action_mask)
                opponent_agent.train(opponent_action_mask)

            state = next_state

        env.state.print_turn_events(file_path=log_path)

        # Update metrics
        metrics['episode_rewards'].append(episode_reward)
        metrics['episode_lengths'].append(steps)
        metrics['player_epsilon'].append(player_agent.epsilon)
        metrics['opponent_epsilon'].append(opponent_agent.epsilon)

        # Update win counters
        if episode_reward > 0:
            metrics['player_wins'] += 1
        elif episode_reward < 0:
            metrics['opponent_wins'] += 1

        # Calculate moving average
        window_size = 100
        if len(metrics['episode_rewards']) >= window_size:
            moving_avg = np.mean(metrics['episode_rewards'][-window_size:])
        else:
            moving_avg = np.mean(metrics['episode_rewards'])
        metrics['moving_avg_reward'].append(moving_avg)

        env.state.print_turn_events(file_path=log_path)
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(f"\nEpisode {episode + 1} ended with reward: {reward}\n")
            f.write(f"Steps taken: {steps}\n")
            f.write('-' * 20 + '\n')

        # Update epsilon for exploration
        player_agent.update_epsilon()
        opponent_agent.update_epsilon()

        # Periodically update target networks
        if episode % target_update == 0:
            player_agent.update_target_network()
            opponent_agent.update_target_network()
    # Save models
    torch.save(player_agent.policy_net.state_dict(), out_dir / 'player_model.pth')
    torch.save(opponent_agent.policy_net.state_dict(), out_dir / 'opponent_model.pth')

    # Save metrics
    with open(out_dir / 'metrics.json', 'w') as f:
        json.dump(metrics, f)

    # Generate and save plots
    create_training_plots(metrics, out_dir)

def create_training_plots(metrics, out_dir):
    # Episode Rewards
    plt.figure(figsize=(10, 6))
    plt.plot(metrics['episode_rewards'], label='Episode Reward', alpha=0.3)
    plt.plot(metrics['moving_avg_reward'], label='Moving Average', linewidth=2)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Training Rewards Over Time')
    plt.legend()
    plt.savefig(out_dir / 'rewards.png')
    plt.close()

    # Episode Lengths
    plt.figure(figsize=(10, 6))
    plt.plot(metrics['episode_lengths'])
    plt.xlabel('Episode')
    plt.ylabel('Steps')
    plt.title('Episode Lengths Over Time')
    plt.savefig(out_dir / 'episode_lengths.png')
    plt.close()


if __name__ == "__main__":
    main(episodes=1000)
