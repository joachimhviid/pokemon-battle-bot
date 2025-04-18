from typing import Any, TypeAlias, cast

import gymnasium as gym
import math
import random
import numpy as np

from pokemon import Pokemon, PokemonMove, PokemonStatKey, PokemonType
from pokemon_parser import parse_team
from battle_effects_manager import BattleEffectsManager
from pokemon_types import NonVolatileStatusCondition, Side, VolatileStatusCondition, WeatherType, is_barrier, is_field, \
    is_hazard, is_terrain, is_valid_boost_stage
from pokemon_utils import get_stat_modifier

# ObsType = dict[str, Union[np.integer, list[np.integer]]]
ObsType = dict[str, Any]
ActType: TypeAlias = int


class BattleEnv(gym.Env[ObsType, ActType]):
    turn_counter: int = 0
    turn_events: dict[int, list[str]] = {}

    battle_effects_manager: BattleEffectsManager

    player_1_team: list[Pokemon]
    player_2_team: list[Pokemon]

    battle_field: dict[Side, list[Pokemon]] = {'player_1': [], 'player_2': []}

    def __init__(self, player_1_team: list[Pokemon], player_2_team: list[Pokemon]):
        MAX_PLAYER_SWITCH_OPTIONS = 6
        MAX_PLAYER_MOVES = 4
        MAX_MOVE_TARGETS = 2
        # TODO: implement masking and map integer to action
        self.action_space = gym.spaces.Discrete(MAX_PLAYER_MOVES * MAX_MOVE_TARGETS + MAX_PLAYER_SWITCH_OPTIONS)
        self.observation_space = gym.spaces.Dict({
            'player_1_active_pokemon': gym.spaces.Box(0, 1, shape=(11,), dtype=np.float32),
            'player_1_team': gym.spaces.Box(0, 1, shape=(6, 11), dtype=np.float32),
            'player_1_fields': gym.spaces.MultiBinary(5),
            'player_1_hazards': gym.spaces.MultiBinary(4),
            'player_1_barriers': gym.spaces.MultiBinary(3),

            'player_2_active_pokemon': gym.spaces.Box(0, 1, shape=(11,), dtype=np.float32),
            'player_2_team': gym.spaces.Box(0, 1, shape=(6, 11), dtype=np.float32),
            'player_2_fields': gym.spaces.MultiBinary(5),
            'player_2_hazards': gym.spaces.MultiBinary(4),
            'player_2_barriers': gym.spaces.MultiBinary(3),

            'weather': gym.spaces.Discrete(5),
            'terrain': gym.spaces.Discrete(5),
        })
        self.player_1_team = player_1_team
        self.player_2_team = player_2_team

        self.battle_field['player_1'].append(player_1_team[0])
        self.battle_field['player_2'].append(player_2_team[0])

        self.battle_effects_manager = BattleEffectsManager()

        for pkm in self.battle_field['player_1'] + self.battle_field['player_2']:
            pkm.on_switch_in()

    def action_to_move(self) -> dict[ActType, tuple[PokemonMove, Pokemon] | Pokemon]:
        """
        Should return a tuple of a move and a target or a switch to another Pokémon
        """
        return {
            0: (self.battle_field['player_1'][0].moves[0], self.battle_field['player_1'][0]),
            1: (self.battle_field['player_1'][0].moves[0], self.battle_field['player_2'][0]),
            2: (self.battle_field['player_1'][0].moves[1], self.battle_field['player_1'][0]),
            3: (self.battle_field['player_1'][0].moves[1], self.battle_field['player_2'][0]),
            4: (self.battle_field['player_1'][0].moves[2], self.battle_field['player_1'][0]),
            5: (self.battle_field['player_1'][0].moves[2], self.battle_field['player_2'][0]),
            6: (self.battle_field['player_1'][0].moves[3], self.battle_field['player_1'][0]),
            7: (self.battle_field['player_1'][0].moves[3], self.battle_field['player_2'][0]),
            8: self.player_1_team[0],
            9: self.player_1_team[1],
            10: self.player_1_team[2],
            11: self.player_1_team[3],
            12: self.player_1_team[4],
            13: self.player_1_team[5],
        }

    def step(self, action: ActType):  # type: ignore
        _action = self.action_to_move()[action]
        if isinstance(_action, tuple):
            move, _ = _action
            self.battle_field['player_1'][0].selected_move = move

        self.turn_counter += 1
        speed_sorted_pokemon = self.battle_field['player_1'] + self.battle_field['player_2']
        speed_sorted_pokemon.sort(key=lambda pkm: pkm.get_boosted_stat('speed'), reverse=True)

        self.on_turn_start(speed_sorted_pokemon)

        if isinstance(_action, tuple):
            move, target = _action
            for pkm in self.get_turn_order():
                print(f'{pkm.name} acts')
                self.execute_move(move=move, attacker=pkm, target=target)
        else:
            for pkm in self.get_turn_order():
                print(f'{pkm.name} acts')
                self.switch_pokemon(side=self.get_pokemon_side(pkm), selected_pokemon=_action)

        self.on_turn_end(speed_sorted_pokemon)

        terminated = self.get_winner() is not None
        truncated = False
        reward = 1 if self.get_winner() == 'player_1' else -1 if self.get_winner() == 'player_2' else 0
        observation = self._get_obs()
        # info = self._get_info()
        return observation, reward, terminated, truncated, {}  # type: ignore

    def get_winner(self) -> Side | None:
        player_1_fainted = all(pokemon.is_fainted() for pokemon in self.player_1_team)
        player_2_fainted = all(pokemon.is_fainted() for pokemon in self.player_2_team)
        return 'player_1' if player_1_fainted else 'player_2' if player_2_fainted else None


    def on_turn_start(self, sorted_active_pokemon: list[Pokemon]):
        for pkm in sorted_active_pokemon:
            pkm.on_turn_start()

    def on_turn_end(self, sorted_active_pokemon: list[Pokemon]):
        # apply field effects (sandstorm chip, grassy terrain healing)
        # decrement field effect counters (weather, terrain, etc)
        # apply status effect (burn/poison damage)
        # trigger items (leftovers)
        for pkm in sorted_active_pokemon:
            pkm.on_turn_end()
        self.battle_effects_manager.on_turn_end(sorted_active_pokemon)

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        # Reset the HP, status effects, stat boosts and restore all PP to Pokemon on each team.
        # Set active Pokemon back to first in list
        for pokemon in self.player_1_team:
            pokemon.reset()
        for pokemon in self.player_2_team:
            pokemon.reset()
        self.battle_effects_manager.reset()
        observation = self._get_obs()
        # info = self._get_info()

        return observation, {}

    def _get_obs(self) -> ObsType:
        """
        The current state as seen by an agent.
        """
        obs = {
            'player_1_active_pokemon': self.battle_field['player_1'][0].encode(),
            'player_1_team': self.encode_team(self.player_1_team),
            'player_1_fields': self.battle_effects_manager.encode_fields('player_1'),
            'player_1_barriers': self.battle_effects_manager.encode_barriers('player_1'),
            'player_1_hazards': self.battle_effects_manager.encode_hazards('player_1'),

            'player_2_active_pokemon': self.battle_field['player_2'][0].encode(),
            'player_2_team': self.encode_team(self.player_2_team),
            'player_2_fields': self.battle_effects_manager.encode_fields('player_2'),
            'player_2_barriers': self.battle_effects_manager.encode_barriers('player_2'),
            'player_2_hazards': self.battle_effects_manager.encode_hazards('player_2'),

            'weather': self.battle_effects_manager.encode_weather(),
            'terrain': self.battle_effects_manager.encode_terrain(),
        }
        self.validate_observation(obs)
        return obs

    def validate_observation(self, observation: ObsType):
        assert self.observation_space.contains(observation), 'Invalid observation'

    def encode_team(self, team: list[Pokemon]) -> np.ndarray[Any, np.dtype[np.float32]]:
        team_vecs = [pkm.encode() for pkm in team]
        # Pad with zeros if fewer than 6 Pokémon
        while len(team_vecs) < 6:
            team_vecs.append(np.zeros(11, dtype=np.float32))
        return np.stack(team_vecs[:6])

    # def _get_info(self):
    #     return {
    #         'turn_number': self.turn_counter,
    #         'events': self.turn_events[self.turn_counter],  # Each players action and outcomes for the turn
    #     }

    def get_turn_order(self) -> list[Pokemon]:
        active_pokemon = self.battle_field['player_1'] + self.battle_field['player_2']
        active_pokemon.sort(key=lambda pkm: (pkm.selected_move.priority, pkm.get_boosted_stat('speed')), reverse=True)
        if active_pokemon[0].stats['speed'] == active_pokemon[1].stats['speed'] and active_pokemon[
            0].selected_move.priority == active_pokemon[1].selected_move.priority:
            random.shuffle(active_pokemon)
        return active_pokemon

    def switch_pokemon(self, side: Side, selected_pokemon: Pokemon):
        match side:
            case 'player_1':
                non_active = [pkm for pkm in self.player_1_team if
                              pkm is not self.battle_field['player_1'][0] and not pkm.is_fainted()]
                if non_active:
                    self.battle_field['player_1'][0].on_switch_out()
                    self.battle_field['player_1'][0] = selected_pokemon
                    self.battle_field['player_1'][0].on_switch_in()
            case 'player_2':
                non_active = [pkm for pkm in self.player_2_team if
                              pkm is not self.battle_field['player_2'][0] and not pkm.is_fainted()]
                if non_active:
                    self.battle_field['player_2'][0].on_switch_out()
                    self.battle_field['player_2'][0] = selected_pokemon
                    self.battle_field['player_2'][0].on_switch_in()

    def execute_move(self, move: PokemonMove, attacker: Pokemon, target: Pokemon):
        if attacker.is_incapacitated():
            self._log_event(f'{attacker.name} was unable to execute move')
            return
        self._log_event(f'{attacker.name} used {move.name} on {target.name}')
        # TODO: Handle incapacitation (flinch, full para, recharge, infatuation)
        move.current_pp = move.current_pp - 1
        inflicted_damage: int = 0
        restored_health: int = 0
        if move.target in ['all-other-pokemon', 'random-opponent', 'selected-pokemon',
                           'all-opponents'] and target.protected:
            self._log_event(f'{target.name} protected itself')
            return
        if not self.is_hit(move, attacker, target):
            self._log_event(f'{target.name} avoided the attack')
            return
        match move.category:
            case 'ailment':
                if move.ailment_is_volatile():
                    target.apply_volatile_status(cast(VolatileStatusCondition, move.ailment_type))
                elif target.non_volatile_status_condition is None:
                    target.apply_non_volatile_status(cast(NonVolatileStatusCondition, move.ailment_type))
            case 'damage':
                inflicted_damage = self.calculate_damage(move, attacker, target)
                self._log_event(f'{target.name} took {inflicted_damage} damage')
            case 'damage+ailment':
                inflicted_damage = self.calculate_damage(move, attacker, target)
                self._log_event(f'{target.name} took {inflicted_damage} damage')
                if random.randint(1, 100) <= move.ailment_chance:
                    if move.ailment_is_volatile():
                        target.apply_volatile_status(cast(VolatileStatusCondition, move.ailment_type))
                    elif target.non_volatile_status_condition is None:
                        target.apply_non_volatile_status(cast(NonVolatileStatusCondition, move.ailment_type))
            case 'net-good-stats':
                self.boost_stat(move, target)
            case 'heal':
                restored_health = math.floor(attacker.stats['hp'] * (move.healing / 100))
            case 'swagger':
                if move.ailment_is_volatile():
                    target.apply_volatile_status(cast(VolatileStatusCondition, move.ailment_type))
                elif target.non_volatile_status_condition is None:
                    target.apply_non_volatile_status(cast(NonVolatileStatusCondition, move.ailment_type))
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
                if move.name == 'rain-dance':
                    self.battle_effects_manager.set_weather('rain')
                if move.name == 'sunny-day':
                    self.battle_effects_manager.set_weather('sunshine')
                if move.name in ['snowscape', 'chilly-reception']:
                    self.battle_effects_manager.set_weather('snow')
                if move.name == 'sandstorm':
                    self.battle_effects_manager.set_weather('sandstorm')
                if is_terrain(move.name):
                    self.battle_effects_manager.set_terrain(move.name)
                if move.name == 'haze':
                    self.battle_field['player_1'][0].reset_boosts()
                    self.battle_field['player_2'][0].reset_boosts()
            case 'field-effect':
                if is_field(move.name):
                    self.battle_effects_manager.add_field_effect(move.name, self.get_pokemon_side(target))
                if is_barrier(move.name):
                    self.battle_effects_manager.add_barrier(move.name, self.get_pokemon_side(target))
                if is_hazard(move.name):
                    self.battle_effects_manager.add_hazard(move.name, self.get_pokemon_side(target))
            case 'force-switch':
                raise NotImplementedError("Force-switch moves are not supported yet.")
            case 'unique':
                if move.name == 'protect':
                    target.protected = True

        # TODO: handle two-turn moves, recharge moves
        target.take_damage(inflicted_damage)
        attacker.restore_health(restored_health)
        # Flinching
        if random.randint(1, 100) <= move.flinch_chance and target.ability != 'inner-focus':
            target.flinched = True
        if target.is_fainted():
            self._log_event(f'{target.name} fainted')
        # Recoil
        if move.drain < 0:
            attacker.take_damage(int(inflicted_damage * (move.drain / 100)))
            self._log_event(f'{attacker.name} was damaged by the recoil')
        if attacker.is_fainted():
            self._log_event(f'{attacker.name} fainted')

    def get_pokemon_side(self, pokemon: Pokemon) -> Side:
        # I'll believe this works when I see it
        if pokemon in self.player_1_team:
            return 'player_1'
        elif pokemon in self.player_2_team:
            return 'player_2'
        else:
            raise ValueError("The given Pokemon does not belong to either team.")

    def boost_stat(self, move: PokemonMove, target: Pokemon):
        for stat_change in move.stat_changes:
            for stat, change in stat_change.items():
                boost = max(-6, min(6, target.stat_boosts[stat] + change))
                if is_valid_boost_stage(boost):
                    target.stat_boosts[stat] = boost
                    self._log_event(f"{target.name}'s {stat} changed by {change} stages")

    def is_hit(self, move: PokemonMove, attacker: Pokemon, defender: Pokemon) -> bool:
        if move.accuracy is None:
            return True
        attacker_accuracy = attacker.stat_boosts['accuracy']
        defender_evasion = defender.stat_boosts['evasion']
        acc_evasion_delta = attacker_accuracy - defender_evasion
        accuracy_threshold = move.accuracy
        if is_valid_boost_stage(acc_evasion_delta):
            accuracy_threshold = move.accuracy * get_stat_modifier(acc_evasion_delta)
        to_hit_threshold = random.randint(1, 100)
        return accuracy_threshold > to_hit_threshold

    def calculate_damage(self, move: PokemonMove, attacker: Pokemon, defender: Pokemon) -> int:
        """https://bulbapedia.bulbagarden.net/wiki/Damage#Generation_V_onward"""
        stab_modifier = 2 if move.type in attacker.types and attacker.ability == 'adaptability' else 1.5 if move.type in attacker.types else 1
        is_crit = self.is_critical_hit(move, attacker)
        if is_crit:
            self._log_event("It's a critical hit!")
        offensive_effective_stat, defensive_effective_stat = self.get_effective_stats(
            attacker=attacker, defender=defender, move=move, is_crit=is_crit)
        # We only model 1v1 battles so this will always be 1
        targets_modifier = 1
        weather_modifier = self.get_weather_modifier(move)
        random_modifier = random.randint(85, 101) / 100
        type_modifier = self.get_type_effectiveness(move, defender)
        if type_modifier == 0:
            self._log_event("It didn't effect the opposing Pokemon")
        elif type_modifier < 1:
            self._log_event("It's not very effective")
        elif type_modifier > 1:
            self._log_event("It's super effective")
        burn_modifier = 0.5 if move.damage_class == 'physical' and attacker.non_volatile_status_condition and attacker.non_volatile_status_condition.name == 'burn' and attacker.ability != 'guts' and move.name != 'facade' else 1
        # Item, ability, aura boosts etc (eg choice band)
        other_modifier = 1
        return int(math.floor((((((2 * attacker.level) / 5) + 2) * (move.power if move.power else 0) * (
                offensive_effective_stat / defensive_effective_stat)) / 50) + 2) * targets_modifier * weather_modifier * (
                       1.5 if is_crit else 1) * random_modifier * stab_modifier * type_modifier * burn_modifier * other_modifier)

    def get_effective_stats(self, move: PokemonMove, attacker: Pokemon, defender: Pokemon, is_crit: bool) -> \
            tuple[float, float]:
        # TODO: Handle cases where effective stat differs from standard (eg Psyshock effective attacking stat is special but deals physical damage)
        def get_offensive_stat(pkm: Pokemon, stat: PokemonStatKey) -> float:
            """Returns the effective offensive stat, considering critical hit mechanics."""
            if is_crit and pkm.stat_boosts[stat] < 0:
                return pkm.stats[stat]
            return pkm.get_boosted_stat(stat)

        def get_defensive_stat(pkm: Pokemon, stat: PokemonStatKey, weather: WeatherType,
                               affected_type: PokemonType) -> float:
            """Returns the effective defensive stat, considering weather conditions and critical hit mechanics."""
            base = pkm.stats[stat] if (is_crit and pkm.stat_boosts[stat] > 0) else pkm.get_boosted_stat(stat)
            if self.battle_effects_manager.weather is not None and weather == self.battle_effects_manager.weather.name and affected_type in defender.types:
                return base * 1.5
            return base

        match move.damage_class:
            case 'physical':
                offensive_stat = get_offensive_stat(attacker, 'attack')
                defensive_stat = get_defensive_stat(defender, 'defense', 'snow', 'ice')
            case 'special':
                offensive_stat = get_offensive_stat(attacker, 'special-attack')
                defensive_stat = get_defensive_stat(defender, 'special-defense', 'sandstorm', 'rock')
            case _:
                return 0, 0  # Fallback case if an unknown damage class is encountered

        return offensive_stat, defensive_stat

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
        type_chart: dict[PokemonType, dict[PokemonType, float]] = {
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
        for defender_type in defender.types:
            effectiveness *= type_chart.get(attacking_move.type, {}).get(defender_type, 1)
        return effectiveness

    def get_weather_modifier(self, move: PokemonMove) -> float:
        if self.battle_effects_manager.weather is None:
            return 1.0
        match self.battle_effects_manager.weather.name:
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
            case _:
                return 1.0
        return 1.0

    def _log_event(self, battle_event: str):
        if self.turn_counter not in self.turn_events:
            self.turn_events[self.turn_counter] = []
        self.turn_events[self.turn_counter].append(battle_event)


if __name__ == "__main__":
    team_1 = parse_team('player_1')
    team_2 = parse_team('player_2')
    env = BattleEnv(team_1, team_2)
    print(f'Team 1 first pokemon {env.battle_field["player_1"][0].name}')
    print(f'Team 2 first pokemon {env.battle_field["player_2"][0].name}')
    for i in range(10):
        env.execute_move(
            env.battle_field["player_1"][0].moves[3], env.battle_field["player_1"][0], env.battle_field["player_2"][0])
    for event in env.turn_events[0]:
        print(event)
