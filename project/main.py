import gymnasium as gym
from tqdm import tqdm

from project import BattleAgent, BattleEnv, team_1, team_2


def main(episodes: int = 1000, target_update: int = 10, log_name: str = 'test-log.txt'):
    # env = gym.make("Pokemon-v0")
    env = BattleEnv(player_team=team_1, opponent_team=team_2)

    player_agent = BattleAgent(env)
    opponent_agent = BattleAgent(env)

    env.player_agent = player_agent
    env.opponent_agent = opponent_agent

    # Clear the log file at the start of a new run
    with open(log_name, 'w', encoding='utf-8') as f:
        f.write("Pokemon Battle Training Log\n\n")

    # Training loop
    for episode in tqdm(range(episodes)):
        state, _ = env.reset()
        done = False

        # Write episode header
        with open(log_name, 'a', encoding='utf-8') as f:
            f.write(f"\nEpisode {episode + 1}\n{'=' * 20}\n")

        while not done:

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

        env.state.print_turn_events(file_path=log_name)
        # Add episode result
        with open(log_name, 'a', encoding='utf-8') as f:
            f.write(f"\nEpisode {episode + 1} ended with reward: {reward}\n")
            f.write('-' * 20 + '\n')

        # Update epsilon for exploration
        player_agent.update_epsilon()
        opponent_agent.update_epsilon()

        # Periodically update target networks
        if episode % target_update == 0:
            player_agent.update_target_network()
            opponent_agent.update_target_network()


if __name__ == "__main__":
    main(episodes=100)
