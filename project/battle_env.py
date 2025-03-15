from typing import Literal
import gymnasium as gym
import math
import random

from pokemon import Pokemon, PokemonMove, PokemonStatBoostStage
from pokemon_parser import parse_team

TerrainType = Literal['grassy-terrain',
                      'electric-terrain', 'misty-terrain', 'psychic-terrain']
WeatherType = Literal['sunshine', 'rain', 'snow', 'sandstorm']


class BattleEnv(gym.Env):
    done: bool = False

    player_1_team: list[Pokemon]
    player_2_team: list[Pokemon]
    player_1_active_pokemon: Pokemon
    player_2_active_pokemon: Pokemon

    terrain: TerrainType | None = None
    weather: WeatherType | None = None

    def __init__(self, player_1_team: list[Pokemon], player_2_team: list[Pokemon]):
        # TODO: set action space to game mechanics (fight: dict of moves, switch: dict of team members)
        self.action_space = gym.spaces.Discrete(4)
        # TODO: set observation space to visible game info (enemy hp, type, etc)
        self.observation_space = gym.spaces.Discrete(4)
        self.player_1_team = player_1_team
        self.player_1_active_pokemon = player_1_team[0]
        self.player_2_team = player_2_team
        self.player_2_active_pokemon = player_2_team[0]

    def step(self):
        pass

    def reset(self):
        # Reset the HP, status effects, stat boosts and restore all PP to Pokemon on each team.
        # Set active Pokemon back to first in list
        for pokemon in self.player_1_team:
            pokemon.reset()
        for pokemon in self.player_2_team:
            pokemon.reset()
        self.done = False

    def get_state(self):
        """
        The state as seen by an agent. We assume standard competetive team sheet knowledge.
        - species info (types)
        - current HP
        - current status conditions
        - current stat boosts
        - list of moves
        - held items
        - ability
        """
        return {
            'player_1_hp': self.player_1_active_pokemon.stats['hp']
        }

    def is_hit(self, move: PokemonMove, attacker: Pokemon, defender: Pokemon) -> bool:
        if move.accuracy == None:
            return True
        # Protect logic here?
        attacker_accuracy = self.get_stat_modifier(
            attacker.stat_boosts['accuracy'])
        defender_evasion = self.get_stat_modifier(
            defender.stat_boosts['evasion'])
        accuracy_threshold = move.accuracy * \
            (attacker_accuracy - defender_evasion)
        to_hit_threshold = random.randint(1, 100)
        return accuracy_threshold > to_hit_threshold

    def calculate_damage(self, move: PokemonMove, attacker: Pokemon, defender: Pokemon) -> int:
        """https://bulbapedia.bulbagarden.net/wiki/Damage#Generation_V_onward"""
        stab_modifier = 2 if move.type in attacker.types and attacker.ability == 'adaptability' else 1.5 if move.type in attacker.types else 1
        is_critical_hit = self.is_critical_hit(move, attacker)
        offensive_effective_stat, defensive_effective_stat = self.get_effective_stats(
            attacker=attacker, defender=defender, move=move, is_critical_hit=is_critical_hit)
        # We only model 1v1 battles so this will always be 1
        targets_modifier = 1
        weather_modifier = self.get_weather_modifier(move)
        random_modifier = random.randint(85, 100) / 100
        type_modifier = self.get_type_effectiveness(move, defender)
        burn_modifier = 0.5 if move.damage_class == 'physical' and attacker.non_volatile_status_condition == 'burn' and attacker.ability != 'guts' and move.name != 'facade' else 1
        # Item boosts etc (eg choice band)
        other_modifier = 1

        return int(math.floor(((((2 * attacker.level) / 5) * move.power * (offensive_effective_stat / defensive_effective_stat)) / 50) + 2) * targets_modifier * weather_modifier * (1.5 if is_critical_hit else 1) * random_modifier * stab_modifier * type_modifier * burn_modifier * other_modifier)

    def is_critical_hit(self, move: PokemonMove, attacker: Pokemon) -> bool:
        match attacker.crit_stage + move.crit_rate:
            case 0:
                return random.randint(1, 24) == 1
            case 1:
                return random.randint(1, 8) == 1
            case 2:
                return random.randint(1, 2) == 1
            case 3:
                return True
            case _:
                return True

    def get_type_effectiveness(self, attacking_move: PokemonMove, defender: Pokemon) -> float:
        effectiveness = 1.0
        type_chart = {
            'normal': {'rock': 0.5, 'ghost': 0, 'steel': 0.5},
            'fire': {'fire': 0.5, 'water': 0.5, 'grass': 2, 'ice': 2, 'bug': 2, 'rock': 0.5, 'dragon': 0.5, 'steel': 2},
            'water': {'fire': 2, 'water': 0.5, 'grass': 0.5, 'ground': 2, 'rock': 2, 'dragon': 0.5},
            'electric': {'water': 2, 'electric': 0.5, 'grass': 0.5, 'ground': 0, 'flying': 2, 'dragon': 0.5},
            'grass': {'fire': 0.5, 'water': 2, 'grass': 0.5, 'poison': 0.5, 'ground': 2, 'flying': 0.5, 'bug': 0.5, 'rock': 2, 'dragon': 0.5, 'steel': 0.5},
            'ice': {'fire': 0.5, 'water': 0.5, 'grass': 2, 'ice': 0.5, 'ground': 2, 'flying': 2, 'dragon': 2, 'steel': 0.5},
            'fighting': {'normal': 2, 'ice': 2, 'poison': 0.5, 'flying': 0.5, 'psychic': 0.5, 'bug': 0.5, 'rock': 2, 'ghost': 0, 'dark': 2, 'steel': 2, 'fairy': 0.5},
            'poison': {'grass': 2, 'poison': 0.5, 'ground': 0.5, 'rock': 0.5, 'ghost': 0.5, 'steel': 0, 'fairy': 2},
            'ground': {'fire': 2, 'electric': 2, 'grass': 0.5, 'poison': 2, 'flying': 0, 'bug': 0.5, 'rock': 2, 'steel': 2},
            'flying': {'electric': 0.5, 'grass': 2, 'fighting': 2, 'bug': 2, 'rock': 0.5, 'steel': 0.5},
            'psychic': {'fighting': 2, 'poison': 2, 'psychic': 0.5, 'dark': 0, 'steel': 0.5},
            'bug': {'fire': 0.5, 'grass': 2, 'fighting': 0.5, 'poison': 0.5, 'flying': 0.5, 'psychic': 2, 'ghost': 0.5, 'dark': 2, 'steel': 0.5, 'fairy': 0.5},
            'rock': {'fire': 2, 'ice': 2, 'fighting': 0.5, 'ground': 0.5, 'flying': 2, 'bug': 2, 'steel': 0.5},
            'ghost': {'normal': 0, 'psychic': 2, 'ghost': 2, 'dark': 0.5},
            'dragon': {'dragon': 2, 'steel': 0.5, 'fairy': 0},
            'dark': {'fighting': 0.5, 'psychic': 2, 'ghost': 2, 'dark': 0.5, 'fairy': 0.5},
            'steel': {'fire': 0.5, 'water': 0.5, 'electric': 0.5, 'ice': 2, 'rock': 2, 'steel': 0.5, 'fairy': 2},
            'fairy': {'fire': 0.5, 'fighting': 2, 'poison': 0.5, 'dragon': 2, 'dark': 2, 'steel': 0.5}
        }
        for defender_type in defender.types:
            effectiveness *= type_chart.get(attacking_move.type,
                                            {}).get(defender_type, 1)
        return effectiveness

    def get_stat_modifier(self, stat_stage: PokemonStatBoostStage) -> float:
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

    def get_effective_stats(self, move: PokemonMove, attacker: Pokemon, defender: Pokemon, is_critical_hit: bool) -> tuple[float, float]:
        # TODO: Handle cases where effective stat differs from standard (eg Psyshock effective attacking stat is special but deals physical damage)
        offensive_effective_stat: float = 0
        defensive_effective_stat: float = 0
        match move.damage_class:
            case 'physical':
                if is_critical_hit:
                    if attacker.stat_boosts['attack'] < 0:
                        offensive_effective_stat = attacker.stats['attack']
                    else:
                        offensive_effective_stat = attacker.stats['attack'] * self.get_stat_modifier(
                            attacker.stat_boosts['attack'])
                    if defender.stat_boosts['defense'] > 0:
                        if self.weather == 'snow' and 'ice' in defender.types:
                            defensive_effective_stat = defender.stats['defense'] * 1.5
                        else:
                            defensive_effective_stat = defender.stats['defense']
                    else:
                        if self.weather == 'snow' and 'ice' in defender.types:
                            defensive_effective_stat = defender.stats['defense'] * self.get_stat_modifier(
                                defender.stat_boosts['defense']) * 1.5
                        else:
                            defensive_effective_stat = defender.stats['defense'] * self.get_stat_modifier(
                                defender.stat_boosts['defense'])
                else:
                    offensive_effective_stat = attacker.stats['attack'] * self.get_stat_modifier(
                        attacker.stat_boosts['attack'])
                    if self.weather == 'snow' and 'ice' in defender.types:
                        defensive_effective_stat = defender.stats['defense'] * self.get_stat_modifier(
                            defender.stat_boosts['defense']) * 1.5
                    else:
                        defensive_effective_stat = defender.stats['defense'] * self.get_stat_modifier(
                            defender.stat_boosts['defense'])
            case 'special':
                if is_critical_hit:
                    if attacker.stat_boosts['special-attack'] < 0:
                        offensive_effective_stat = attacker.stats['special-attack']
                    else:
                        offensive_effective_stat = attacker.stats['special-attack'] * self.get_stat_modifier(
                            attacker.stat_boosts['special-attack'])
                    if defender.stat_boosts['special-defense'] > 0:
                        if self.weather == 'sandstorm' and 'rock' in defender.types:
                            defensive_effective_stat = defender.stats['special-defense'] * 1.5
                        else:
                            defensive_effective_stat = defender.stats['special-defense']
                    else:
                        if self.weather == 'sandstorm' and 'rock' in defender.types:
                            defensive_effective_stat = defender.stats['special-defense'] * self.get_stat_modifier(
                                defender.stat_boosts['special-defense']) * 1.5
                        else:
                            defensive_effective_stat = defender.stats['special-defense'] * self.get_stat_modifier(
                                defender.stat_boosts['special-defense'])
                else:
                    offensive_effective_stat = attacker.stats['special-attack'] * self.get_stat_modifier(
                        attacker.stat_boosts['special-attack'])
                    if self.weather == 'sandstorm' and 'rock' in defender.types:
                        defensive_effective_stat = defender.stats['special-defense'] * self.get_stat_modifier(
                            defender.stat_boosts['special-defense']) * 1.5
                    else:
                        defensive_effective_stat = defender.stats['special-defense'] * self.get_stat_modifier(
                            defender.stat_boosts['special-defense'])

        return offensive_effective_stat, defensive_effective_stat
    
    def get_weather_modifier(self, move: PokemonMove) -> float:
        match self.weather:
            case 'rain':
                if move.type == 'water':
                    return 1.5
                elif move.type == 'fire':
                    return 0.5
                else:
                    return 1
            case 'sunshine':
                if move.type == 'fire':
                    return 1.5
                elif move.type == 'water':
                    return 0.5
                else:
                    return 1
            case _:
                return 1
                


if __name__ == "__main__":
    team_1 = parse_team('player_1')
    team_2 = parse_team('player_2')
    env = BattleEnv(team_1, team_2)
    print(f'Team 1 first pokemon {env.player_1_active_pokemon.name}')
    print(f'Team 2 first pokemon {env.player_2_active_pokemon.name}')
    for i in range(10):
        print(f'{env.player_1_active_pokemon.name} uses {env.player_1_active_pokemon.moves[3].name} on {env.player_2_active_pokemon.name} dealing {env.calculate_damage(
            env.player_1_active_pokemon.moves[3], env.player_1_active_pokemon, env.player_2_active_pokemon)} damage')
