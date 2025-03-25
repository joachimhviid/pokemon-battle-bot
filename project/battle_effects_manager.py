from typing import Literal, TypedDict

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
Weather = TypedDict('Weather', {'name': WeatherType, 'duration': int})
Terrain = TypedDict('Terrain', {'name': TerrainType, 'duration': int})
BiasedEffect = TypedDict('BiasedEffect', {
                         'player_1': dict[BarrierType | HazardType | FieldType, int], 'player_2': dict[BarrierType | HazardType | FieldType, int]})
Side = Literal['player_1', 'player_2']


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
                self.hazards[side][hazard_type] = self.hazards[side][hazard_type] + 1
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

    def end_turn(self):
        def reduce_fields(side: Side):
            for field in self.fields[side]:
                if self.fields[side][field] == 0:
                    self.fields[side].pop(field)
                else:
                    self.fields[side][field] = self.fields[side][field] - 1

        reduce_fields('player_1')
        reduce_fields('player_2')

        def reduce_barriers(side: Side):
            for barrier in self.barriers[side]:
                if self.barriers[side][barrier] == 0:
                    self.barriers[side].pop(barrier)
                else:
                    self.barriers[side][barrier] = self.barriers[side][barrier] - 1

        reduce_barriers('player_1')
        reduce_barriers('player_2')

        if self.terrain is not None:
            if self.terrain['duration'] == 0:
                self.terrain = None
            else:
                self.terrain['duration'] = self.terrain['duration'] - 1

        if self.weather is not None:
            if self.weather['duration'] == 0:
                self.weather = None
            else:
                self.weather['duration'] = self.weather['duration'] - 1

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
