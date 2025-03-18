from typing import Literal, Union
import gymnasium as gym
import math
import random

from pokemon import Pokemon, PokemonMove, PokemonStatBoostStage, PokemonType
from pokemon_parser import parse_team

TerrainType = Literal['grassy-terrain',
                      'electric-terrain', 'misty-terrain', 'psychic-terrain']
WeatherType = Literal['sunshine', 'rain', 'snow', 'sandstorm']
FieldType = Literal[
    'mist',
    'light-screen',
    'reflect',
    'spikes',
    'safeguard',
    'tailwind',
    'lucky-chant',
    'toxic-spikes',
    'stealth-rock',
    'wide-guard',
    'quick-guard',
    'mat-block',
    'sticky-web',
    'crafty-shield',
    'aurora-veil'
]
BattleEffect = Union[TerrainType, WeatherType, FieldType]


class BattleEnv(gym.Env):
    turn_counter: int = 0
    turn_events: dict[int, list[str]] = {}

    player_1_team: list[Pokemon]
    player_2_team: list[Pokemon]
    player_1_active_pokemon: Pokemon
    player_2_active_pokemon: Pokemon

    terrain: TerrainType | None = None
    weather: WeatherType | None = None
    battle_effects: dict[BattleEffect, int] = {}

    def __init__(self, player_1_team: list[Pokemon], player_2_team: list[Pokemon]):
        # TODO: set action space to game mechanics (fight: dict of moves, switch: dict of team members)
        self.action_space = gym.spaces.Discrete(4)
        # TODO: set observation space to visible game info (enemy hp, type, etc)
        self.observation_space = gym.spaces.Discrete(4)
        self.player_1_team = player_1_team
        self.player_1_active_pokemon = player_1_team[0]
        self.player_2_team = player_2_team
        self.player_2_active_pokemon = player_2_team[0]

    def step(self, action: str):
        # Get action from dict based on arg
        # Apply action to environment (Execute move)
        # Determine if battle is over (terminated)
        # Truncate the environment if it is too slow?
        # Reward agent
        print('step')
        terminated = False
        truncated = False
        reward = 0
        observation = self._get_obs()
        info = self._get_info()
        return observation, reward, terminated, truncated, info

    def reset(self):
        # Reset the HP, status effects, stat boosts and restore all PP to Pokemon on each team.
        # Set active Pokemon back to first in list
        for pokemon in self.player_1_team:
            pokemon.reset()
        for pokemon in self.player_2_team:
            pokemon.reset()

    def _get_obs(self):
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
            'player_1_active_pokemon': self.player_1_active_pokemon,
            'player_1_team': self.player_1_team,
            'player_2_active_pokemon': self.player_2_active_pokemon,
            'player_2_team': self.player_2_team,
        }

    def _get_info(self):
        return {
            'turn_number': self.turn_counter,
            'events': self.turn_events[self.turn_counter],  # Each players action and outcomes for the turn
        }

    def execute_move(self, move: PokemonMove, attacker: Pokemon, target: Pokemon):
        self._log_event(f'{attacker.name} used {move.name} on {target.name}')
        move.current_pp = move.current_pp - 1
        inflicted_damage: int = 0
        restored_health: int = 0
        if not self.is_hit(move, attacker, target):
            self._log_event(f'{target.name} avoided the attack')
            return
        match move.category:
            case 'ailment':
                # TODO: handle known durations such as sleep
                if move.ailment_is_volatile():
                    target.volatile_status_condition.update({
                        move.ailment_type: -1 if move.duration['min'] == None
                        else random.randint(move.duration['min'], move.duration['max'])
                    })
                elif len(target.non_volatile_status_condition.keys()) == 0:
                    target.non_volatile_status_condition = {
                        move.ailment_type: -1 if move.duration['min'] == None
                        else random.randint(move.duration['min'], move.duration['max'])
                    }
            case 'damage':
                inflicted_damage = self.calculate_damage(move, attacker, target)
                self._log_event(f'{target.name} took {inflicted_damage} damage')
            case 'damage+ailment':
                inflicted_damage = self.calculate_damage(move, attacker, target)
                self._log_event(f'{target.name} took {inflicted_damage} damage')
                if random.randint(1, 100) <= move.ailment_chance:
                    if move.ailment_is_volatile():
                        target.volatile_status_condition.update({
                            move.ailment_type: -1 if move.duration['min'] == None
                            else random.randint(move.duration['min'], move.duration['max'])
                        })
                    elif len(target.non_volatile_status_condition.keys()) == 0:
                        target.non_volatile_status_condition = {
                            move.ailment_type: -1 if move.duration['min'] == None
                            else random.randint(move.duration['min'], move.duration['max'])
                        }
            case 'net-good-stats':
                self.boost_stat(move, target)
            case 'heal':
                restored_health = math.floor(attacker.stats['hp'] * (move.healing / 100))
            case 'swagger':
                if move.ailment_is_volatile():
                    target.volatile_status_condition.update({
                        move.ailment_type: -1 if move.duration['min'] == None
                        else random.randint(move.duration['min'], move.duration['max'])
                    })
                elif len(target.non_volatile_status_condition.keys()) == 0:
                    target.non_volatile_status_condition = {
                        move.ailment_type: -1 if move.duration['min'] == None
                        else random.randint(move.duration['min'], move.duration['max'])
                    }
                self.boost_stat(move, target)
            case 'damage+lower':
                inflicted_damage = self.calculate_damage(move, attacker, target)
                self._log_event(f'{target.name} took {inflicted_damage} damage')
                if random.randint(1, 100) <= move.stat_chance:
                    self.boost_stat(move, target)
            case 'damage+raise':
                inflicted_damage = self.calculate_damage(move, attacker, target)
                self._log_event(f'{target.name} took {inflicted_damage} damage')
                if random.randint(1, 100) <= move.stat_chance:
                    self.boost_stat(move, attacker)
            case 'damage+heal':
                inflicted_damage = self.calculate_damage(move, attacker, target)
                self._log_event(f'{target.name} took {inflicted_damage} damage')
                restored_health = math.floor(inflicted_damage * (move.drain / 100))
            case 'ohko':
                inflicted_damage = target.current_hp
                self._log_event("It's a one-hit KO!")
            case 'whole-field-effect':
                raise NotImplementedError("Whole-field-effect moves are not supported yet.")
            case 'field-effect':
                raise NotImplementedError("Field-effect moves are not supported yet.")
            case 'force-switch':
                raise NotImplementedError("Force-switch moves are not supported yet.")
            case 'unique':
                raise NotImplementedError("Unique moves are not supported yet.")

        target.current_hp = target.current_hp - inflicted_damage
        attacker.current_hp = min(attacker.current_hp + restored_health, attacker.stats['hp'])
        if target.is_fainted():
            self._log_event(f'{target.name} fainted')

    def boost_stat(self, move: PokemonMove, target: Pokemon):
        for stat_change in move.stat_changes:
            for stat, change in stat_change.items():
                target.stat_boosts[stat] = max(-6, min(6, target.stat_boosts[stat] + change))
                self._log_event(f"{target.name}'s {stat} changed by {change} stages")

    def is_hit(self, move: PokemonMove, attacker: Pokemon, defender: Pokemon) -> bool:
        if move.accuracy == None:
            return True
        # Protect logic here?
        attacker_accuracy = self.get_stat_modifier(
            attacker.stat_boosts['accuracy'])
        defender_evasion = self.get_stat_modifier(
            defender.stat_boosts['evasion'])
        accuracy_threshold = move.accuracy * self.get_stat_modifier(attacker_accuracy - defender_evasion)
        to_hit_threshold = random.randint(1, 100)
        return accuracy_threshold > to_hit_threshold

    def calculate_damage(self, move: PokemonMove, attacker: Pokemon, defender: Pokemon) -> int:
        """https://bulbapedia.bulbagarden.net/wiki/Damage#Generation_V_onward"""
        stab_modifier = 2 if move.type in attacker.types and attacker.ability == 'adaptability' else 1.5 if move.type in attacker.types else 1
        is_critical_hit = self.is_critical_hit(move, attacker)
        if is_critical_hit:
            print("It's a critical hit!")
        offensive_effective_stat, defensive_effective_stat = self.get_effective_stats(
            attacker=attacker, defender=defender, move=move, is_critical_hit=is_critical_hit)
        # We only model 1v1 battles so this will always be 1
        targets_modifier = 1
        weather_modifier = self.get_weather_modifier(move)
        random_modifier = random.randint(85, 100) / 100
        type_modifier = self.get_type_effectiveness(move, defender)
        if type_modifier == 0:
            print("It didn't effect the opposing Pokemon")
        elif type_modifier < 1:
            print("It's not very effective")
        elif type_modifier > 1:
            print("It's super effective")
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
        def get_offensive_stat(base_stat: float, boost: PokemonStatBoostStage) -> float:
            """Returns the effective offensive stat, considering critical hit mechanics."""
            if is_critical_hit and boost < 0:
                return base_stat
            return base_stat * self.get_stat_modifier(boost)

        def get_defensive_stat(base_stat: float, boost: PokemonStatBoostStage, weather: WeatherType, affected_type: PokemonType) -> float:
            """Returns the effective defensive stat, considering weather conditions and critical hit mechanics."""
            base = base_stat if (is_critical_hit and boost >
                                 0) else base_stat * self.get_stat_modifier(boost)
            if weather == self.weather and affected_type in defender.types:
                return base * 1.5
            return base

        match move.damage_class:
            case  'physical':
                offensive_stat = get_offensive_stat(
                    attacker.stats['attack'], attacker.stat_boosts['attack'])
                defensive_stat = get_defensive_stat(
                    defender.stats['defense'], defender.stat_boosts['defense'], 'snow', 'ice')
            case 'special':
                offensive_stat = get_offensive_stat(
                    attacker.stats['special-attack'], attacker.stat_boosts['special-attack'])
                defensive_stat = get_defensive_stat(
                    defender.stats['special-defense'], defender.stat_boosts['special-defense'], 'sandstorm', 'rock')
            case _:
                return 0, 0  # Fallback case if an unknown damage class is encountered

        return offensive_stat, defensive_stat

    def get_weather_modifier(self, move: PokemonMove) -> float:
        # TODO: is_grounded check for terrains. We should also make sure this is the correct modifier for terrain buffs
        for effect in self.battle_effects.keys():
            match effect:
                case 'rain':
                    if move.type == 'water':
                        return 1.5
                    elif move.type == 'fire':
                        return 0.5
                case 'sunshine':
                    if move.type == 'fire':
                        return 1.5
                    elif move.type == 'water':
                        return 0.5
                case 'electric-terrain':
                    if move.type == 'electric':
                        return 1.3
                case 'grassy-terrain':
                    if move.type == 'grass':
                        return 1.3
                case 'misty-terrain':
                    if move.type == 'dragon':
                        return 0.5
                case 'psychic-terrain':
                    if move.type == 'psychic':
                        return 1.3
                case _:
                    return 1
            return 1
        
    def _log_event(self, event: str):
        self.turn_events[self.turn_counter].append(event)


if __name__ == "__main__":
    team_1 = parse_team('player_1')
    team_2 = parse_team('player_2')
    env = BattleEnv(team_1, team_2)
    print(f'Team 1 first pokemon {env.player_1_active_pokemon.name}')
    print(f'Team 2 first pokemon {env.player_2_active_pokemon.name}')
    env.execute_move(
        env.player_1_active_pokemon.moves[2], env.player_1_active_pokemon, env.player_1_active_pokemon)
    # for i in range(10):
    #     print(f'{env.player_1_active_pokemon.name} uses {env.player_1_active_pokemon.moves[3].name} on {env.player_2_active_pokemon.name} dealing {env.calculate_damage(
    #         env.player_1_active_pokemon.moves[3], env.player_1_active_pokemon, env.player_2_active_pokemon)} damage')
