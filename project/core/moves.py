from dataclasses import dataclass, field
from typing import Optional, Literal, Any

from project.utils.constants import PokemonBoostStatKey, PokemonStatBoostStage, MoveAilment, MoveCategory, MoveTarget, \
    DamageClass, PokemonType, VolatileStatusCondition


@dataclass(init=False)
class PokemonMove:
    name: str
    power: Optional[int]
    type: PokemonType
    pp: int
    current_pp: int
    accuracy: Optional[int]
    damage_class: DamageClass
    priority: int
    target: MoveTarget
    category: MoveCategory

    ailment_type: MoveAilment
    ailment_chance: int

    stat_changes: list[dict[PokemonBoostStatKey, PokemonStatBoostStage]]
    stat_chance: int

    healing: int
    flinch_chance: int
    drain: int
    crit_rate: int

    hits: dict[Literal['min', 'max'], Optional[int]] = field(default_factory=lambda: {
        'min': None,
        'max': None,
    })
    duration: dict[Literal['min', 'max'], Optional[int]] = field(default_factory=lambda: {
        'min': None,
        'max': None,
    })

    def __init__(self, move: Any):
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
        self.hits = {
            'min': move['meta']['min_hits'],
            'max': move['meta']['max_hits']
        }
        self.duration = {
            'min': move['meta']['min_turns'],
            'max': move['meta']['max_turns']
        }
        self.healing = move['meta']['healing']
        self.flinch_chance = move['meta']['flinch_chance']
        self.drain = move['meta']['drain']
        self.crit_rate = move['meta']['crit_rate']

    def ailment_is_volatile(self) -> bool:
        return self.ailment_type in VolatileStatusCondition.__args__