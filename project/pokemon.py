import random
from typing import Literal
import math
from pokemon_utils import get_stat_modifier
from pokemon_types import PokemonNature, PokemonNatureKey, PokemonStatKey, PokemonType, DamageClass, MoveTarget, MoveCategory, MoveAilment, PokemonBoostStatKey, PokemonStatBoostStage, VolatileStatusCondition, NonVolatileStatusCondition, PokemonStats


pokemon_natures: PokemonNature = {
    'hardy': {},
    'lonely': {'UP': 'attack', 'DOWN': 'defense'},
    'brave': {'UP': 'attack', 'DOWN': 'speed'},
    'adamant': {'UP': 'attack', 'DOWN': 'special-attack'},
    'naughty': {'UP': 'attack', 'DOWN': 'special-defense'},
    'bold': {'UP': 'defense', 'DOWN': 'attack'},
    'docile': {},
    'relaxed': {'UP': 'defense', 'DOWN': 'speed'},
    'impish': {'UP': 'defense', 'DOWN': 'special-attack'},
    'lax': {'UP': 'defense', 'DOWN': 'special-defense'},
    'timid': {'UP': 'speed', 'DOWN': 'attack'},
    'hasty': {'UP': 'speed', 'DOWN': 'defense'},
    'serious': {},
    'jolly': {'UP': 'speed', 'DOWN': 'special-attack'},
    'naive': {'UP': 'speed', 'DOWN': 'special-defense'},
    'modest': {'UP': 'special-attack', 'DOWN': 'attack'},
    'mild': {'UP': 'special-attack', 'DOWN': 'defense'},
    'quiet': {'UP': 'special-attack', 'DOWN': 'speed'},
    'bashful': {},
    'rash': {'UP': 'special-attack', 'DOWN': 'special-defense'},
    'calm': {'UP': 'special-defense', 'DOWN': 'attack'},
    'gentle': {'UP': 'special-defense', 'DOWN': 'defense'},
    'sassy': {'UP': 'special-defense', 'DOWN': 'speed'},
    'careful': {'UP': 'special-defense', 'DOWN': 'special-attack'},
    'quirky': {}
}


class PokemonMove:
    name: str
    power: int | None
    type: PokemonType
    pp: int
    current_pp: int
    accuracy: int | None
    damage_class: DamageClass
    priority: int
    target: MoveTarget
    category: MoveCategory

    ailment_type: MoveAilment
    ailment_chance: int

    stat_changes: list[dict[PokemonBoostStatKey, PokemonStatBoostStage]]
    stat_chance: int

    hits: dict[Literal['min', 'max'], int | None] = {
        'min': None,
        'max': None,
    }
    duration: dict[Literal['min', 'max'], int | None] = {
        'min': None,
        'max': None,
    }

    healing: int
    flinch_chance: int
    drain: int
    crit_rate: int

    def __init__(self, move):
        self.name = move['name']
        self.power = move['power']
        self.type = move['type']
        self.pp = move['pp']
        self.current_pp = self.pp
        self.accuracy = move['accuracy']
        self.damage_class = move['damage_class']
        self.priority = move['priority']
        self.target = move['target']
        self.category = move['meta']['category']['name']
        self.ailment_type = move['meta']['ailment']['name']
        self.ailment_chance = move['meta']['ailment_chance']
        self.stat_changes = [
            {change['stat']['name']: change['change']}
            for change in move['stat_changes']
        ]
        self.stat_chance = move['meta']['stat_chance']
        self.hits['max'] = move['meta']['max_hits']
        self.hits['min'] = move['meta']['min_hits']
        self.duration['max'] = move['meta']['max_turns']
        self.duration['min'] = move['meta']['min_turns']
        self.healing = move['meta']['healing']
        self.flinch_chance = move['meta']['flinch_chance']
        self.drain = move['meta']['drain']
        self.crit_rate = move['meta']['crit_rate']

    def ailment_is_volatile(self) -> bool:
        return self.ailment_type in VolatileStatusCondition.__args__


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
    previous_move: PokemonMove | None
    types: list[PokemonType]
    ability: str
    current_hp: int
    crit_stage: int = 0
    stat_boosts: dict[PokemonBoostStatKey, PokemonStatBoostStage] = {
        'attack': 0,
        'defense': 0,
        'special-attack': 0,
        'special-defense': 0,
        'speed': 0,
        'accuracy': 0,
        'evasion': 0
    }
    non_volatile_status_condition: dict[NonVolatileStatusCondition, int] = {}
    volatile_status_condition: dict[VolatileStatusCondition, int] = {}

    held_item: str | None

    # Pokemons action is cancelled (full paralysis, freeze, flinch, etc)
    incapacitated: bool = False
    protected: bool = False
    active: bool = False

    def __init__(self, pokemon_data):
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
        self.ability = pokemon_data['ability']
        self.types = pokemon_data['types']
        self.held_item = pokemon_data['held_item']

    def get_nature_modifier(self, stat: PokemonStatKey) -> float:
        if pokemon_natures.get(self._nature).get('UP') == stat:
            return 1.1
        elif pokemon_natures.get(self._nature).get('DOWN') == stat:
            return 0.9
        else:
            return 1.0

    def _calculate_stat_value(self, stat: PokemonStatKey) -> int:
        nature_modifier: float = self.get_nature_modifier(stat)
        stat_value = self._base_stats.get(stat)
        iv_value = self._ivs.get(stat)
        ev_value = self._evs.get(stat)

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
                self.non_volatile_status_condition[status] = random.randint(1, 3)
            case 'poison' | 'bad-poison' | 'burn' | 'paralysis' | 'freeze':
                self.non_volatile_status_condition[status] = -1

    def apply_volatile_status(self, status: VolatileStatusCondition):
        match status:
            case 'confusion':
                self.volatile_status_condition[status] = random.randint(2, 5)
            case 'trap':
                self.volatile_status_condition[status] = random.randint(4, 5)
            case 'disable':
                self.volatile_status_condition[status] = 4
            case 'encore':
                self.volatile_status_condition[status] = 3
            case 'yawn':
                self.volatile_status_condition[status] = 1
            case 'leech-seed' | 'infatuation' | 'ingrain':
                self.volatile_status_condition[status] = -1

    def get_boosted_stat(self, stat: PokemonStatKey) -> int:
        if stat == 'hp':
            return self.stats['hp']
        return self.stats[stat] * get_stat_modifier(self.stat_boosts[stat])

    def on_turn_start(self):
        for duration in self.volatile_status_condition.values():
            duration -= 1
        for duration in self.non_volatile_status_condition.values():
            duration -= 1

    def on_turn_end(self):
        self.protected = False
        for status in self.non_volatile_status_condition.keys():
            match status:
                case 'poison':
                    self.current_hp -= self.stats['hp'] // 8
                case 'bad-poison':
                    self.current_hp -= self.stats['hp'] * abs(self.non_volatile_status_condition['bad-poison']) // 16
                case 'burn':
                    self.current_hp -= self.stats['hp'] // 16
        for status in self.volatile_status_condition.keys():
            match status:
                case 'leech-seed':
                    # TODO: Handle leeching effect. Somehow get the damage taken from here and add it to the opposing pokemon
                    self.current_hp -= self.stats['hp'] // 8
                case 'ingrain':
                    self.current_hp = min(self.current_hp + self.stats['hp'] // 16, self.stats['hp'])
                case 'yawn':
                    if self.volatile_status_condition['yawn'] == 0 and len(self.non_volatile_status_condition.keys()) == 0:
                        self.apply_non_volatile_status('sleep')
                case 'trap':
                    self.current_hp -= self.stats['hp'] // 8

    def on_switch_out(self):
        self.volatile_status_condition.clear()
        if 'bad-poison' in self.non_volatile_status_condition:
            self.non_volatile_status_condition['bad-poison'] = -1
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


if __name__ == "__main__":
    print('pokemon')
    max_hp = 200
    current = max_hp
    for counter in [-1, -2, -3, -4, -5]:
        dt = max_hp * abs(counter) // 16
        print(dt)
