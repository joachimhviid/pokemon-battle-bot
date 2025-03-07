from typing import Literal
import math

PokemonNatureKey = Literal[
    'hardy', 'lonely', 'brave', 'adamant', 'naughty', 'bold', 'docile', 'relaxed', 'impish', 'lax', 'timid', 'hasty', 'serious', 'jolly', 'naive', 'modest', 'mild', 'quiet', 'bashful', 'rash', 'calm', 'gentle', 'sassy', 'careful', 'quirky']
PokemonStatKey = Literal['hp', 'attack',
                         'defense', 'special-attack', 'special-defense', 'speed']
PokemonNatureModifier = Literal['UP', 'DOWN']
PokemonStats = dict[PokemonStatKey, int]
PokemonNature = dict[
    PokemonNatureKey, dict[PokemonNatureModifier, PokemonStatKey]]
PokemonType = Literal[
    'normal', 'fire', 'water', 'electric', 'grass', 'ice', 'fighting', 'poison', 'ground', 'flying', 'psychic', 'bug', 'rock', 'ghost', 'dragon', 'dark', 'steel', 'fairy']
DamageClass = Literal['status', 'special', 'physical']
MoveTarget = Literal['specific-move', 'selected-pokemon-me-first', 'ally', 'users-field', 'user-or-ally', 'opponents-field', 'user', 'random-opponent', 'all-other-pokemon'
                     'selected-pokemon', 'all-opponents', 'entire-field', 'user-and-allies', 'all-pokemon', 'all-allies', 'fainting-pokemon']
MoveCategory = Literal[
    'damage', 'ailment', 'net-good-stats', 'heal', 'damage+ailment', 'swagger',
    'damage+lower', 'damage+raise', 'damage+heal', 'ohko', 'whole-field-effect',
    'field-effect', 'force-switch', 'unique'
]
MoveAilment = Literal[
    'unknown', 'none', 'paralysis', 'sleep', 'freeze', 'burn', 'poison', 'confusion', 'infatuation', 'trap', 'nightmare', 'torment', 'disable', 'yawn', 'heal-block', 'no-type-immunity', 'leech-seed', 'embargo', 'perish-song', 'ingrain'
]

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
    accuracy: int | None
    damage_class: DamageClass
    priority: int
    target: MoveTarget
    category: MoveCategory

    ailment_type: MoveAilment
    ailment_chance: int

    stat_changes: list[dict[PokemonStatKey, int]]
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
        self.accuracy = move['accuracy']
        self.damage_class = move['damage_class']
        self.priority = move['priority']
        self.target = move['target']
        self.category = move['meta']['category']['name']
        self.ailment_type = move['meta']['ailment']['name']
        self.ailment_chance = move['meta']['ailment_chance']
        self.stat_changes = move['stat_changes']
        self.stat_chance = move['meta']['stat_chance']
        self.hits['max'] = move['meta']['max_hits']
        self.hits['min'] = move['meta']['min_hits']
        self.duration['max'] = move['meta']['max_turns']
        self.duration['min'] = move['meta']['min_turns']
        self.healing = move['meta']['healing']
        self.flinch_chance = move['meta']['flinch_chance']
        self.drain = move['meta']['drain']
        self.crit_rate = move['meta']['crit_rate']


class Pokemon:
    name: str
    _base_stats: PokemonStats
    _ivs: PokemonStats
    _evs: PokemonStats
    _nature: PokemonNatureKey
    _level: int
    stats: PokemonStats
    moves: list[PokemonMove]

    def __init__(self, pokemon_data):
        self.name = pokemon_data['name']
        self._base_stats = pokemon_data['stats']
        self._ivs = pokemon_data['ivs']
        self._evs = pokemon_data['evs']
        self._nature = pokemon_data['nature']
        self._level = pokemon_data['level']
        self.stats = {stat: self._calculate_stat_value(
            stat) for stat in self._base_stats}
        self.moves = [PokemonMove(move) for move in pokemon_data['moves']]

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
            return math.floor(((2 * stat_value + iv_value + ev_value // 4) * self._level // 100) + self._level + 10)

        return math.floor((((2 * stat_value + iv_value + ev_value // 4) * self._level // 100) + 5) * nature_modifier)


if __name__ == "__main__":
    print('pokemon')
