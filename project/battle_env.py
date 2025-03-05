import gymnasium as gym


class BattleEnv(gym.env):
    def __init__(self):
        # TODO: set action space to game mechanics (fight: dict of moves, switch: dict of team members)
        self.action_space = gym.spaces.Discrete(4)
        # TODO: set observation space to visible game info (enemy hp, type, etc)
        self.observation_space = gym.spaces.Discrete(4)

    def step():
        pass

    def reset():
        pass


def main():
    pass


if __name__ == "__main__":
    main()
