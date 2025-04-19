from typing import Literal, Union, Dict

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

POKEMON_NATURES: PokemonNature = {
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

TYPE_CHART: Dict[PokemonType, Dict[PokemonType, float]] = {
        'normal': {'rock': 0.5, 'ghost': 0, 'steel': 0.5},
        'fire': {'fire': 0.5, 'water': 0.5, 'grass': 2, 'ice': 2, 'bug': 2, 'rock': 0.5, 'dragon': 0.5, 'steel': 2},
        'water': {'fire': 2, 'water': 0.5, 'grass': 0.5, 'ground': 2, 'rock': 2, 'dragon': 0.5},
        'electric': {'water': 2, 'electric': 0.5, 'grass': 0.5, 'ground': 0, 'flying': 2, 'dragon': 0.5},
        'grass': {'fire': 0.5, 'water': 2, 'grass': 0.5, 'poison': 0.5, 'ground': 2, 'flying': 0.5, 'bug': 0.5,
                  'rock': 2, 'dragon': 0.5, 'steel': 0.5},
        'ice': {'fire': 0.5, 'water': 0.5, 'grass': 2, 'ice': 0.5, 'ground': 2, 'flying': 2, 'dragon': 2,
                'steel': 0.5},
        'fighting': {'normal': 2, 'ice': 2, 'poison': 0.5, 'flying': 0.5, 'psychic': 0.5, 'bug': 0.5, 'rock': 2,
                     'ghost': 0, 'dark': 2, 'steel': 2, 'fairy': 0.5},
        'poison': {'grass': 2, 'poison': 0.5, 'ground': 0.5, 'rock': 0.5, 'ghost': 0.5, 'steel': 0, 'fairy': 2},
        'ground': {'fire': 2, 'electric': 2, 'grass': 0.5, 'poison': 2, 'flying': 0, 'bug': 0.5, 'rock': 2,
                   'steel': 2},
        'flying': {'electric': 0.5, 'grass': 2, 'fighting': 2, 'bug': 2, 'rock': 0.5, 'steel': 0.5},
        'psychic': {'fighting': 2, 'poison': 2, 'psychic': 0.5, 'dark': 0, 'steel': 0.5},
        'bug': {'fire': 0.5, 'grass': 2, 'fighting': 0.5, 'poison': 0.5, 'flying': 0.5, 'psychic': 2, 'ghost': 0.5,
                'dark': 2, 'steel': 0.5, 'fairy': 0.5},
        'rock': {'fire': 2, 'ice': 2, 'fighting': 0.5, 'ground': 0.5, 'flying': 2, 'bug': 2, 'steel': 0.5},
        'ghost': {'normal': 0, 'psychic': 2, 'ghost': 2, 'dark': 0.5},
        'dragon': {'dragon': 2, 'steel': 0.5, 'fairy': 0},
        'dark': {'fighting': 0.5, 'psychic': 2, 'ghost': 2, 'dark': 0.5, 'fairy': 0.5},
        'steel': {'fire': 0.5, 'water': 0.5, 'electric': 0.5, 'ice': 2, 'rock': 2, 'steel': 0.5, 'fairy': 2},
        'fairy': {'fire': 0.5, 'fighting': 2, 'poison': 0.5, 'dragon': 2, 'dark': 2, 'steel': 0.5}
    }