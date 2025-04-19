import numpy as np
from typing import Any, Optional, List
from project.core.pokemon import Pokemon
from project.data.parsers import parse_team
from project.effects.biased_effect import BiasedEffect
from project.effects.terrain import Terrain
from project.effects.weather import Weather
from project.utils.constants import WeatherType, TerrainType, HazardType, BarrierType, Side, FieldType


class BattleEffectsManager:
    weather: Optional[Weather]
    terrain: Optional[Terrain]
    barriers: BiasedEffect
    fields: BiasedEffect
    hazards: BiasedEffect

    def __init__(self):
        self.weather = None
        self.terrain = None
        self.barriers = BiasedEffect()
        self.fields = BiasedEffect()
        self.hazards = BiasedEffect()

    def set_weather(self, weather_type: WeatherType, turns: int = 5):
        self.weather = Weather(weather_type, turns)

    def set_terrain(self, terrain_type: TerrainType, turns: int = 5):
        self.terrain = Terrain(terrain_type, turns)

    def add_hazard(self, hazard_type: HazardType, side: Side):
        self.hazards.add_effect(hazard_type, side)

    def add_barrier(self, barrier_type: BarrierType, side: Side, turns: int = 5):
        self.barriers.add_effect(barrier_type, side, turns)

    def clear_barriers(self, side: Side):
        self.barriers.effects_for_side(side).clear()

    def add_field_effect(self, field_effect_type: FieldType, side: Side):
        turns: int = 0
        match field_effect_type:
            case 'mist' | 'safeguard':
                turns = 5
            case 'tailwind':
                turns = 4
            case 'quick-guard' | 'wide-guard':
                turns = 1

        self.fields.add_effect(field_effect_type, side, turns)

    def _reduce_effects(self):
        self.fields.reduce()
        self.barriers.reduce()

    def _handle_terrain(self, active_pokemon: List[Pokemon]):
        if not self.terrain:
            return

        if self.terrain.duration == 0:
            self.terrain = None
            return

        self.terrain.duration -= 1

        if self.terrain.name == 'grassy-terrain':
            for pkm in active_pokemon:
                pkm.restore_health(pkm.stats['hp'] // 16)

    def _handle_weather(self, active_pokemon: List[Pokemon]):
        if not self.weather:
            return

        if self.weather.duration == 0:
            self.weather = None
            return

        self.weather.duration -= 1

        for pkm in active_pokemon:
            if self.weather.name == 'sandstorm' and not any(
                    type_ in pkm.types for type_ in ['rock', 'steel', 'ground']) and pkm.held_item != 'safety-goggles':
                pkm.take_damage(pkm.stats['hp'] // 16)
            if self.weather.name == 'rain':
                if pkm.ability == 'rain-dish':
                    pkm.restore_health(pkm.stats['hp'] // 16)
                if pkm.ability == 'dry-skin':
                    pkm.restore_health(pkm.stats['hp'] // 8)
            if self.weather.name == 'sunshine':
                if pkm.ability == 'dry-skin':
                    pkm.take_damage(pkm.stats['hp'] // 8)
                if pkm.ability == 'solar-power':
                    pkm.take_damage(pkm.stats['hp'] // 8)

    def on_turn_end(self, active_pokemon: List[Pokemon]):
        self._reduce_effects()
        self._handle_terrain(active_pokemon)
        self._handle_weather(active_pokemon)

    def reset(self):
        self.weather = None
        self.terrain = None
        self.barriers.reset()
        self.fields.reset()
        self.hazards.reset()

    def encode_hazards(self, side: Side) -> np.ndarray[Any, np.dtype[np.int8]]:
        return np.array([
            1 if self.hazards.effects_for_side(side)['spikes'] else 0,
            1 if self.hazards.effects_for_side(side)['toxic-spikes'] else 0,
            1 if self.hazards.effects_for_side(side)['stealth-rocks'] else 0,
            1 if self.hazards.effects_for_side(side)['sticky-web'] else 0,
        ], dtype=np.int8)

    def encode_barriers(self, side: Side) -> np.ndarray[Any, np.dtype[np.int8]]:
        return np.array([
            1 if self.barriers.effects_for_side(side)['reflect'] else 0,
            1 if self.barriers.effects_for_side(side)['light-screen'] else 0,
            1 if self.barriers.effects_for_side(side)['aurora-veil'] else 0,
        ], dtype=np.int8)

    def encode_fields(self, side: Side) -> np.ndarray[Any, np.dtype[np.int8]]:
        return np.array([
            1 if self.fields.effects_for_side(side)['mist'] else 0,
            1 if self.fields.effects_for_side(side)['safeguard'] else 0,
            1 if self.fields.effects_for_side(side)['tailwind'] else 0,
            1 if self.fields.effects_for_side(side)['wide-guard'] else 0,
            1 if self.fields.effects_for_side(side)['quick-guard'] else 0,
        ], dtype=np.int8)

    def encode_weather(self) -> int:
        if self.weather is None:
            return 0
        return self.weather.encode()

    def encode_terrain(self) -> int:
        if self.terrain is None:
            return 0
        return self.terrain.encode()


if __name__ == "__main__":
    print('battle effects')
    team_1 = parse_team('player_1')
    if any(type_ in team_1[2].types for type_ in ['rock', 'steel', 'ground']):
        print('present')
