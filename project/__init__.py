from gymnasium.envs.registration import register

from project.data.parsers import parse_team
from project.agent.battle_agent import BattleAgent
from project.battle.battle_env import BattleEnv

# Load teams outside the register to avoid loading them multiple times
team_1 = parse_team('player_1')
team_2 = parse_team('player_2')

register(
    id="Pokemon-v0",
    entry_point="project.battle.battle_env:BattleEnv",
    kwargs={
        "player_team": team_1,
        "opponent_team": team_2
    }
)
