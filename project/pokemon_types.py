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
Side = Literal['player', 'opponent']


@dataclass
class Weather:
    name: WeatherType
    duration: int
    
    def encode(self) -> int:
        match self.name:
            case 'rain':
                return 1
            case 'sunshine':
                return 2
            case 'sandstorm':
                return 3
            case 'snow':
                return 4


@dataclass
class Terrain:
    name: TerrainType
    duration: int
    
    def encode(self) -> int:
        match self.name:
            case 'electric-terrain':
                return 1
            case 'grassy-terrain':
                return 2
            case 'misty-terrain':
                return 3
            case 'psychic-terrain':
                return 4


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
SIDES: tuple[Side, Side] = 'player', 'opponent'


@dataclass
class VolatileStatus:
    name: VolatileStatusCondition
    duration: int

    def encode(self) -> float:
        match self.name:
            case 'confusion':
                return 1.0
            case 'disable':
                return 2.0
            case 'encore':
                return 3.0
            case 'infatuation':
                return 4.0
            case 'ingrain':
                return 5.0
            case 'leech-seed':
                return 6.0
            case 'torment':
                return 7.0
            case 'trap':
                return 8.0
            case 'yawn':
                return 9.0


@dataclass
class NonVolatileStatus:
    name: NonVolatileStatusCondition
    duration: int

    def encode(self) -> float:
        match self.name:
            case 'paralysis':
                return 1.0
            case 'sleep':
                return 2.0
            case 'burn':
                return 3.0
            case 'freeze':
                return 4.0
            case 'poison':
                return 5.0
            case 'bad-poison':
                return 6.0


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

def encode_type(type: PokemonType) -> float:
    match type:
        case 'normal':
            return 1.0
        case 'fire':
            return 2.0
        case 'water':
            return 3.0
        case 'electric':
            return 4.0
        case 'grass':
            return 5.0
        case 'ice':
            return 6.0
        case 'fighting':
            return 7.0
        case 'poison':
            return 8.0
        case 'ground':
            return 9.0
        case 'flying':
            return 10.0
        case 'psychic':
            return 11.0
        case 'bug':
            return 12.0
        case 'rock':
            return 13.0
        case 'ghost':
            return 14.0
        case 'dragon':
            return 15.0
        case 'dark':
            return 16.0
        case 'steel':
            return 17.0
        case 'fairy':
            return 18.0


if __name__ == "__main__":
    print('pokemon types')
