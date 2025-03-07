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


class Pokemon:
    name: str
    _base_stats: PokemonStats
    _ivs: PokemonStats
    _evs: PokemonStats
    _nature: PokemonNatureKey
    _level: int
    stats: PokemonStats
    held_item: str | None
    opponent: True | False

    def __init__(self, name: str, base_stats: PokemonStats, ivs: PokemonStats, evs: PokemonStats,
                 nature: PokemonNatureKey, level: int, held_item: str | None, types: list[str], moves: list[str], ability: str):
        self.name = name
        self._base_stats = base_stats
        self._ivs = ivs
        self._evs = evs
        self._nature = nature
        self._level = level
        self.stats = {stat: self._calculate_stat_value(
            stat) for stat in self._base_stats}
        self.held_item = held_item
        self.opponent = False

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


class PokemonMove:
    name: str
    power: int
    type: PokemonType

    def __init__(self):
        pass
