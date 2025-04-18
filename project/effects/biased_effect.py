from dataclasses import dataclass, field
from typing import Optional

from project.utils.constants import EffectType, Side
from project.utils.type_utils import is_hazard

STACKABLE_HAZARDS = {'spikes': 3, 'toxic-spikes': 2}


@dataclass
class BiasedEffect:
    player: dict[EffectType, int] = field(default_factory=dict)
    opponent: dict[EffectType, int] = field(default_factory=dict)

    def reduce(self):
        for effects in (self.player, self.opponent):
            to_remove: list[EffectType] = []
            for effect, turns in effects.items():
                if turns == 0:
                    to_remove.append(effect)
                else:
                    effects[effect] -= 1
            for effect in to_remove:
                effects.pop(effect)

    def effects_for_side(self, side: Side) -> dict[EffectType, int]:
        return self.player if side == 'player' else self.opponent

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
        self.player.clear()
        self.opponent.clear()


if __name__ == "__main__":
    print('BiasedEffect')
