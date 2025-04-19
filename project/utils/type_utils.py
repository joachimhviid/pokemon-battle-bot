from typing import TypeGuard

from project.utils.constants import PokemonStatBoostStage, PokemonType, TerrainType, HazardType, FieldType, BarrierType


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
    return 0.0

def get_stat_modifier(stat_stage: PokemonStatBoostStage) -> float:
    match stat_stage:
        case 0:
            return 2 / 2
        case 1:
            return 3 / 2
        case 2:
            return 4 / 2
        case 3:
            return 5 / 2
        case 4:
            return 6 / 2
        case 5:
            return 7 / 2
        case 6:
            return 8 / 2
        case -1:
            return 2 / 3
        case -2:
            return 2 / 4
        case -3:
            return 2 / 5
        case -4:
            return 2 / 6
        case -5:
            return 2 / 7
        case -6:
            return 2 / 8
        case _:
            return 1