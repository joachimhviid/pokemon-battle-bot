from gymnasium.envs.registration import register
from .battle_env import BattleEnv

register(
    id="Pokemon-v0",
    entry_point="battle_env:BattleEnv",
)