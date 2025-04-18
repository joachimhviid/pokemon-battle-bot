from dataclasses import dataclass, field
import random
from typing import Any, Optional, Union, cast
import math
import numpy as np

from project.core.moves import PokemonMove

from project.effects.status_effects import VolatileStatus, NonVolatileStatus
from project.utils.constants import POKEMON_NATURES, PokemonStatKey, PokemonBoostStatKey, PokemonType, \
    PokemonStatBoostStage, PokemonStats, PokemonNatureKey, NonVolatileStatusCondition, VolatileStatusCondition, Side, \
    SIDES
from project.utils.type_utils import get_stat_modifier, encode_type


@dataclass(init=False)
class Pokemon:
    name: str
    _base_stats: PokemonStats
    _ivs: PokemonStats
    _evs: PokemonStats
    _nature: PokemonNatureKey
    level: int
    stats: PokemonStats
    moves: list[PokemonMove]
    # Used to determine turn order
    selected_move: PokemonMove
    # Used to determine 2 turn moves, double protect, etc
    previous_move: Optional[PokemonMove]
    types: list[PokemonType]
    ability: str
    current_hp: int
    non_volatile_status_condition: Optional[NonVolatileStatus]

    held_item: Optional[str]

    crit_stage: int = 0
    flinched: bool = False
    protected: bool = False
    active: bool = False
    stat_boosts: dict[Union[PokemonStatKey, PokemonBoostStatKey], PokemonStatBoostStage] = field(default_factory=lambda: {
        'attack': 0,
        'defense': 0,
        'special-attack': 0,
        'special-defense': 0,
        'speed': 0,
        'accuracy': 0,
        'evasion': 0
    })
    volatile_status_conditions: list[VolatileStatus] = field(default_factory=list)

    def __init__(self, pokemon_data: Any):
        self.name = pokemon_data['name']
        self._base_stats = pokemon_data['stats']
        self._ivs = pokemon_data['ivs']
        self._evs = pokemon_data['evs']
        self._nature = pokemon_data['nature']
        self.level = pokemon_data['level']
        self.stats = {stat: self._calculate_stat_value(
            stat) for stat in self._base_stats}
        self.current_hp = self.stats['hp']
        self.moves = [PokemonMove(move) for move in pokemon_data['moves']]
        self.selected_move = self.moves[0]
        self.previous_move = None
        self.ability = pokemon_data['ability']
        self.types = pokemon_data['types']
        self.held_item = pokemon_data['held_item']
        self.non_volatile_status_condition = None
        self.volatile_status_conditions = []
        self.stat_boosts = {
            'attack': 0,
            'defense': 0,
            'special-attack': 0,
            'special-defense': 0,
            'speed': 0,
            'accuracy': 0,
            'evasion': 0
        }

    def get_nature_modifier(self, stat: PokemonStatKey) -> float:
        nature = POKEMON_NATURES.get(self._nature)
        if nature and nature.get('UP') == stat:
            return 1.1
        elif nature and nature.get('DOWN') == stat:
            return 0.9
        else:
            return 1.0

    def _calculate_stat_value(self, stat: PokemonStatKey) -> int:
        nature_modifier: float = self.get_nature_modifier(stat)
        stat_value = self._base_stats.get(stat)
        iv_value = self._ivs.get(stat)
        ev_value = self._evs.get(stat)

        if stat_value is None:
            raise RuntimeError(f'Missing base stats for Pokemon {self.name}')
        if iv_value is None:
            raise RuntimeError(f'Missing ivs for Pokemon {self.name}')
        if ev_value is None:
            raise RuntimeError(f'Missing evs for Pokemon {self.name}')

        if stat == 'hp':
            return math.floor(((2 * stat_value + iv_value + ev_value // 4) * self.level // 100) + self.level + 10)

        return math.floor((((2 * stat_value + iv_value + ev_value // 4) * self.level // 100) + 5) * nature_modifier)

    def take_damage(self, damage: int):
        self.current_hp -= damage

    def restore_health(self, healing: int):
        self.current_hp = min(self.current_hp + healing, self.stats['hp'])

    def reset(self):
        self.reset_boosts()
        self.current_hp = self.stats['hp']
        for move in self.moves:
            move.current_pp = move.pp

    def reset_boosts(self):
        self.stat_boosts = {
            'attack': 0,
            'defense': 0,
            'special-attack': 0,
            'special-defense': 0,
            'speed': 0,
            'accuracy': 0,
            'evasion': 0
        }
        self.crit_stage = 0

    def is_fainted(self) -> bool:
        return self.current_hp <= 0

    def apply_non_volatile_status(self, status: NonVolatileStatusCondition):
        match status:
            case 'sleep':
                self.non_volatile_status_condition = NonVolatileStatus(status, random.randint(1, 3))
            case 'poison' | 'bad-poison' | 'burn' | 'paralysis' | 'freeze':
                self.non_volatile_status_condition = NonVolatileStatus(status, -1)

    def apply_volatile_status(self, status: VolatileStatusCondition):
        match status:
            case 'confusion':
                self.volatile_status_conditions.append(VolatileStatus(status, random.randint(2, 5)))
            case 'trap':
                self.volatile_status_conditions.append(VolatileStatus(status, random.randint(4, 5)))
            case 'disable':
                self.volatile_status_conditions.append(VolatileStatus(status, 4))
            case 'encore':
                self.volatile_status_conditions.append(VolatileStatus(status, 3))
            case 'yawn':
                self.volatile_status_conditions.append(VolatileStatus(status, 1))
            case 'leech-seed' | 'infatuation' | 'ingrain' | 'torment':
                self.volatile_status_conditions.append(VolatileStatus(status, -1))

    def get_boosted_stat(self, stat: PokemonStatKey) -> int:
        if stat == 'hp':
            return self.stats['hp']
        return int(self.stats[stat] * get_stat_modifier(self.stat_boosts[stat]))

    def on_turn_start(self):
        for status in self.volatile_status_conditions:
            status.duration -= 1
            if status.duration == 0:
                self.volatile_status_conditions.remove(status)

        if self.non_volatile_status_condition:
            self.non_volatile_status_condition.duration -= 1
            if self.non_volatile_status_condition.duration == 0:
                self.non_volatile_status_condition = None

    def on_turn_end(self):
        self.protected = False
        if self.non_volatile_status_condition:
            match self.non_volatile_status_condition.name:
                case 'poison':
                    self.current_hp -= self.stats['hp'] // 8
                case 'bad-poison':
                    self.current_hp -= self.stats['hp'] * abs(self.non_volatile_status_condition.duration) // 16
                case 'burn':
                    self.current_hp -= self.stats['hp'] // 16
                case _:
                    pass

        for status in self.volatile_status_conditions:
            match status.name:
                case 'leech-seed':
                    # TODO: Handle leeching effect. Somehow get the damage taken from here and add it to the opposing pokemon
                    self.current_hp -= self.stats['hp'] // 8
                case 'ingrain':
                    self.current_hp = min(self.current_hp + self.stats['hp'] // 16, self.stats['hp'])
                case 'yawn':
                    if status.duration == 0 and self.non_volatile_status_condition is None:
                        self.apply_non_volatile_status('sleep')
                        self.volatile_status_conditions.remove(status)
                case 'trap':
                    self.current_hp -= self.stats['hp'] // 8
                case _:
                    pass

    def on_switch_out(self):
        self.volatile_status_conditions.clear()
        if self.non_volatile_status_condition and self.non_volatile_status_condition.name == 'bad-poison':
            self.non_volatile_status_condition.duration = -1
        self.stat_boosts = {
            'attack': 0,
            'defense': 0,
            'special-attack': 0,
            'special-defense': 0,
            'speed': 0,
            'accuracy': 0,
            'evasion': 0
        }
        self.crit_stage = 0
        self.active = False

    def on_switch_in(self):
        self.active = True

    def is_incapacitated(self) -> bool:
        if self.flinched:
            return True
        for status in [vol_status.name for vol_status in self.volatile_status_conditions] + ([self.non_volatile_status_condition.name] if self.non_volatile_status_condition else []):
            match status:
                case 'confusion':
                    if random.randint(1, 3) == 1:
                        random_modifier = random.randint(85, 101) / 100
                        # 40 base power typeless physical move
                        inflicted_damage = int(math.floor((((((2 * self.level) / 5) + 2) * 40 * (
                            self.stats['attack'] / self.stats['defense'])) / 50) + 2) * random_modifier)
                        self.take_damage(inflicted_damage)
                        return True
                    return False
                case 'infatuation':
                    return True if random.randint(1, 2) == 1 else False
                case 'paralysis':
                    return True if random.randint(1, 4) == 1 else False
                case 'freeze':
                    if random.randint(1, 5) == 1:
                        self.non_volatile_status_condition = None
                        return False
                    return True
                case 'sleep':
                    return True
                case _:
                    return False
        return False

    def encode(self) -> np.ndarray[Any, np.dtype[np.float32]]:
        hp = self.current_hp / self.stats['hp'] if self.stats['hp'] > 0 else 0.0
        non_vol_status = self.non_volatile_status_condition.encode() if self.non_volatile_status_condition else 0.0
        vol_status = [status.encode() for status in self.volatile_status_conditions] if len(
            self.volatile_status_conditions) > 0 else [0.0]
        type_1 = encode_type(self.types[0]) / 18
        type_2 = encode_type(self.types[1]) / 18 if self.types[1] else 0.0

        atk = self.stats['attack'] / 255.0
        def_ = self.stats['defense'] / 255.0
        sp_atk = self.stats['special-attack'] / 255.0
        sp_def = self.stats['special-defense'] / 255.0
        spd = self.stats['speed'] / 255.0
        level = self.level / 100.0
        
        return np.array([
            hp, non_vol_status, vol_status, type_1, type_2, atk, def_, sp_atk, sp_def, spd, level
        ], dtype=np.float32)


# TODO: indicate if move hits all available targets or just one
# TODO: target slot index?
def get_available_targets(user: Pokemon, user_side: Side, move: PokemonMove, active_pokemon: dict[Side, list[Pokemon]]) -> list[list[Pokemon]]:
    opponent_side = cast(Side, [side for side in SIDES if side != user_side][0])
    match move.target:
        # Since we are only modelling single battles there should be no elligible targets here
        case 'ally' | 'all-allies':
            active_pokemon[user_side].remove(user)
            return [active_pokemon[user_side]]
        case 'users-field' | 'user-and-allies':
            return [active_pokemon[user_side]]
        case 'user-or-ally':
            targets: list[list[Pokemon]] = []
            for pkm in active_pokemon[user_side]:
                targets.append([pkm])
            return targets
        case 'opponents-field':
            return [active_pokemon[opponent_side]]
        case 'random-opponent':
            targets = []
            for pkm in active_pokemon[opponent_side]:
                targets.append([pkm])
            return targets
        case 'user':
            return [[user]]
        case 'all-other-pokemon':
            active_pokemon[user_side].remove(user)
            return [active_pokemon[opponent_side] + active_pokemon[user_side]]
        case 'selected-pokemon':
            targets = []
            active_pokemon[user_side].remove(user)
            for pkm in active_pokemon[opponent_side] + active_pokemon[user_side]:
                targets.append([pkm])
            return targets
        case 'all-opponents':
            return [active_pokemon[opponent_side]]
        case 'entire-field' | 'all-pokemon':
            return [active_pokemon[opponent_side] + active_pokemon[user_side]]
        case 'fainting-pokemon':
            raise NotImplementedError('Fainted pokemon target not implemented')
        case 'selected-pokemon-me-first' | 'specific-move':
            raise NotImplementedError('Target type not implemented')


if __name__ == "__main__":
    print('pokemon')
    max_hp = 200
    current = max_hp
    for counter in [-1, -2, -3, -4, -5]:
        dt = max_hp * abs(counter) // 16
        print(dt)
