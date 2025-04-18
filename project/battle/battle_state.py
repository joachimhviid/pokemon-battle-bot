from dataclasses import dataclass

from project.core.pokemon import Pokemon
from project.effects.effects_manager import BattleEffectsManager


@dataclass
class BattleState:
    turn_counter: int
    battle_field: dict[str, list[Pokemon]]
    effects_manager: BattleEffectsManager