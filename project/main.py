import gymnasium as gym
from project.battle.battle_env import BattleEnv
from project.data.parsers import parse_team

def main():
    env = gym.make("Pokemon-v0")
    print(env.action_space)

if __name__ == "__main__":
    main()
