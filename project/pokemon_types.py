from typing import Literal, Union, TypeGuard
from dataclasses import dataclass


WeatherType = Literal['sunshine', 'rain', 'snow', 'sandstorm']
TerrainType = Literal['grassy-terrain', 'electric-terrain', 'misty-terrain', 'psychic-terrain']
BarrierType = Literal['reflect', 'light-screen', 'aurora-veil']
HazardType = Literal['spikes', 'toxic-spikes', 'stealth-rocks', 'sticky-web']
FieldType = Literal[
    'mist',  # 5 turns
    'safeguard',  # 5 turns
    'tailwind',  # 4 turns
    'wide-guard',  # 1 turn
    'quick-guard',  # 1 turn
]
EffectType = Union[BarrierType, HazardType, FieldType]
Side = Literal['player_1', 'player_2']


@dataclass
class Weather:
    name: WeatherType
    duration: int


@dataclass
class Terrain:
    name: TerrainType
    duration: int


PokemonNatureKey = Literal[
    'hardy', 'lonely', 'brave', 'adamant', 'naughty', 'bold', 'docile', 'relaxed', 'impish', 'lax', 'timid', 'hasty', 'serious', 'jolly', 'naive', 'modest', 'mild', 'quiet', 'bashful', 'rash', 'calm', 'gentle', 'sassy', 'careful', 'quirky']
PokemonStatKey = Literal['hp', 'attack',
                         'defense', 'special-attack', 'special-defense', 'speed']
PokemonBoostStatKey = Literal['attack',
                              'defense', 'special-attack', 'special-defense', 'speed', 'accuracy', 'evasion']
PokemonNatureModifier = Literal['UP', 'DOWN']
PokemonStatBoostStage = Literal[-6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6]
PokemonStats = dict[PokemonStatKey, int]
PokemonNature = dict[
    PokemonNatureKey, dict[PokemonNatureModifier, PokemonStatKey]]
PokemonType = Literal[
    'normal', 'fire', 'water', 'electric', 'grass', 'ice', 'fighting', 'poison', 'ground', 'flying', 'psychic', 'bug', 'rock', 'ghost', 'dragon', 'dark', 'steel', 'fairy']
DamageClass = Literal['status', 'special', 'physical']
MoveTarget = Literal['specific-move', 'selected-pokemon-me-first', 'ally', 'users-field', 'user-or-ally', 'opponents-field', 'user', 'random-opponent', 'all-other-pokemon',
                     'selected-pokemon', 'all-opponents', 'entire-field', 'user-and-allies', 'all-pokemon', 'all-allies', 'fainting-pokemon']
MoveCategory = Literal[
    'damage', 'ailment', 'net-good-stats', 'heal', 'damage+ailment', 'swagger',
    'damage+lower', 'damage+raise', 'damage+heal', 'ohko', 'whole-field-effect',
    'field-effect', 'force-switch', 'unique'
]
MoveAilment = Literal[
    'unknown', 'none', 'paralysis', 'sleep', 'freeze', 'burn', 'poison', 'confusion', 'infatuation', 'trap', 'nightmare', 'torment', 'disable', 'yawn', 'heal-block', 'no-type-immunity', 'leech-seed', 'embargo', 'perish-song', 'ingrain'
]
NonVolatileStatusCondition = Literal['paralysis',
                                     'sleep', 'freeze', 'burn', 'poison', 'bad-poison']
VolatileStatusCondition = Literal['confusion', 'infatuation', 'trap', 'torment', 'disable',
                                  'yawn', 'leech-seed', 'ingrain', 'encore']
SIDES: tuple[Side, Side] = 'player_1', 'player_2'


@dataclass
class VolatileStatus:
    name: VolatileStatusCondition
    duration: int


@dataclass
class NonVolatileStatus:
    name: NonVolatileStatusCondition
    duration: int


def is_barrier(value: str) -> TypeGuard[BarrierType]:
    return value in BarrierType.__args__


def is_field(value: str) -> TypeGuard[FieldType]:
    return value in FieldType.__args__

def is_hazard(value: str) -> TypeGuard[HazardType]:
    return value in HazardType.__args__

def is_terrain(value: str) -> TypeGuard[TerrainType]:
    return value in TerrainType.__args__

def is_valid_boost_stage(value: int) -> TypeGuard[PokemonStatBoostStage]:
    return value in PokemonStatBoostStage.__args__


if __name__ == "__main__":
    print('pokemon types')
