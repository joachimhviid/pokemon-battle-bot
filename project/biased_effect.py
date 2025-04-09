from dataclasses import dataclass, field
from typing import Optional

from pokemon_types import EffectType, Side, is_hazard

STACKABLE_HAZARDS = {'spikes': 3, 'toxic-spikes': 2}


@dataclass
class BiasedEffect:
    player_1: dict[EffectType, int] = field(default_factory=dict)
    player_2: dict[EffectType, int] = field(default_factory=dict)

    def reduce(self):
        for effects in (self.player_1, self.player_2):
            to_remove: list[EffectType] = []
            for effect, turns in effects.items():
                if turns == 0:
                    to_remove.append(effect)
                else:
                    effects[effect] -= 1
            for effect in to_remove:
                effects.pop(effect)

    def effects_for_side(self, side: Side) -> dict[EffectType, int]:
        return self.player_1 if side == 'player_1' else self.player_2

    def add_effect(self, effect: EffectType, side: Side, turns: Optional[int] = None):
        if is_hazard(effect):
            hazard_limit = STACKABLE_HAZARDS.get(effect)
            if effect in self.effects_for_side(side) and hazard_limit and hazard_limit > self.effects_for_side(side)[effect]:
                self.effects_for_side(side)[effect] += 1
            else:
                self.effects_for_side(side)[effect] = 1
        elif effect not in self.effects_for_side(side):
            self.effects_for_side(side)[effect] = turns if turns else 5

    def reset(self):
        self.player_1.clear()
        self.player_2.clear()


if __name__ == "__main__":
    print('BiasedEffect')
