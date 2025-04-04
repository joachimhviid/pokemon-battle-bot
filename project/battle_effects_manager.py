from pokemon import Pokemon
from pokemon_parser import parse_team
from pokemon_types import BarrierType, BiasedEffect, FieldType, HazardType, Side, Terrain, TerrainType, Weather, WeatherType


class BattleEffectsManager:
    weather: Weather | None = None
    terrain: Terrain | None = None
    barriers: BiasedEffect = {'player_1': {}, 'player_2': {}}
    fields: BiasedEffect = {'player_1': {}, 'player_2': {}}
    hazards: BiasedEffect = {'player_1': {}, 'player_2': {}}

    def set_weather(self, weather_type: WeatherType, turns: int = 5):
        self.weather = {
            'name': weather_type,
            'duration': turns,
        }

    def set_terrain(self, terrain_type: TerrainType, turns: int = 5):
        self.terrain = {
            'name': terrain_type,
            'duration': turns,
        }

    def add_hazard(self, hazard_type: HazardType, side: Side):
        stackable_hazards = {'spikes': 3, 'toxic-spikes': 2}
        if hazard_type in self.hazards[side] and hazard_type in stackable_hazards:
            if self.hazards[side][hazard_type] != stackable_hazards[hazard_type]:
                self.hazards[side][hazard_type] += 1
        else:
            self.hazards[side][hazard_type] = 1

    def add_barrier(self, barrier_type: BarrierType, side: Side, turns: int = 5):
        if barrier_type not in self.barriers[side]:
            self.barriers[side][barrier_type] = turns

    def clear_barriers(self, side: Side):
        self.barriers[side].clear()

    def add_field_effect(self, field_effect_type: FieldType, side: Side, turns: int):
        if field_effect_type not in self.barriers[side]:
            self.fields[side][field_effect_type] = turns

    def on_turn_end(self, active_pokemon: list[Pokemon]):
        def reduce_fields(side: Side):
            for field in self.fields[side]:
                if self.fields[side][field] == 0:
                    self.fields[side].pop(field)
                else:
                    self.fields[side][field] -= 1

        reduce_fields('player_1')
        reduce_fields('player_2')

        def reduce_barriers(side: Side):
            for barrier in self.barriers[side]:
                if self.barriers[side][barrier] == 0:
                    self.barriers[side].pop(barrier)
                else:
                    self.barriers[side][barrier] -= 1

        reduce_barriers('player_1')
        reduce_barriers('player_2')

        if self.terrain is not None:
            if self.terrain['duration'] == 0:
                self.terrain = None
            else:
                self.terrain['duration'] -= 1

            if self.terrain['name'] == 'grassy-terrain':
                for pkm in active_pokemon:
                    pkm.restore_health(pkm.stats['hp'] // 16)

        if self.weather is not None:
            if self.weather['duration'] == 0:
                self.weather = None
            else:
                self.weather['duration'] -= 1

            for pkm in active_pokemon:
                if self.weather['name'] == 'sandstorm' and not any(type_ in pkm.types for type_ in ['rock', 'steel', 'ground']) and pkm.held_item != 'safety-goggles':
                    pkm.take_damage(pkm.stats['hp'] // 16)
                if self.weather['name'] == 'rain':
                    if pkm.ability == 'rain-dish':
                        pkm.restore_health(pkm.stats['hp'] // 16)
                    if pkm.ability == 'dry-skin':
                        pkm.restore_health(pkm.stats['hp'] // 8)
                if self.weather['name'] == 'sunshine':
                    if pkm.ability == 'dry-skin':
                        pkm.take_damage(pkm.stats['hp'] // 8)
                    if pkm.ability == 'solar-power':
                        pkm.take_damage(pkm.stats['hp'] // 8)

    def is_barrier(self, value: str) -> bool:
        return value in BarrierType.__args__

    def is_field(self, value: str) -> bool:
        return value in FieldType.__args__

    def is_hazard(self, value: str) -> bool:
        return value in HazardType.__args__

    def reset(self):
        self.weather = None
        self.terrain = None
        self.barriers = {'player_1': {}, 'player_2': {}}
        self.fields = {'player_1': {}, 'player_2': {}}
        self.hazards = {'player_1': {}, 'player_2': {}}


if __name__ == "__main__":
    print('battle effects')
    team_1 = parse_team('player_1')
    if any(type_ in team_1[2].types for type_ in ['rock', 'steel', 'ground']):
        print('present')
