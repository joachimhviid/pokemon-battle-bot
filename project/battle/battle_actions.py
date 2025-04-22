import random
import math
from typing import cast
from project.core.pokemon import Pokemon, PokemonMove, PokemonType
from project.utils.constants import Side, WeatherType, PokemonStatKey, VolatileStatusCondition, \
    NonVolatileStatusCondition, TYPE_CHART
from project.utils.type_utils import (
    is_terrain, is_field, is_barrier, is_hazard, is_valid_boost_stage,
    get_stat_modifier
)
from .battle_state import BattleState


class BattleActions:
    def __init__(self, battle_state: BattleState):
        self.state = battle_state



    def switch_pokemon(self, side: Side, selected_pokemon: Pokemon):
        if side == 'player':
            self.state.battle_field[0].on_switch_out()
            self.state.battle_field[0] = selected_pokemon
            self.state.battle_field[0].on_switch_in()
        else:
            self.state.battle_field[1].on_switch_out()
            self.state.battle_field[1] = selected_pokemon
            self.state.battle_field[1].on_switch_in()

    def boost_stat(self, move: PokemonMove, target: Pokemon):
        for stat_change in move.stat_changes:
            for stat, change in stat_change.items():
                boost = max(-6, min(6, target.stat_boosts[stat] + change))
                if is_valid_boost_stage(boost):
                    target.stat_boosts[stat] = boost
                    self.state.log_event(f"{target.name}'s {stat} changed by {change} stages")

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

    def execute_move(self, move: PokemonMove, attacker: Pokemon, target: Pokemon):
        if not self._can_execute_move(attacker, move, target):
            return

        inflicted_damage = 0
        restored_health = 0

        match move.category:
            case 'ailment':
                self._handle_ailment_move(move, target)
            case 'damage':
                inflicted_damage = self._handle_damage_move(move, attacker, target)
            case 'damage+ailment':
                inflicted_damage = self._handle_damage_with_ailment_move(move, attacker, target)
            case 'net-good-stats':
                self.boost_stat(move, target)
            case 'heal':
                restored_health = self._calculate_healing(move, attacker)
            case 'swagger':
                self._handle_swagger_move(move, target)
            case 'damage+lower' | 'damage+raise':
                inflicted_damage = self._handle_damage_with_stat_change(move, attacker, target)
            case 'damage+heal':
                inflicted_damage = self._handle_damage_with_healing(move, attacker, target)
                restored_health = math.floor(inflicted_damage * (move.drain / 100))
            case 'ohko':
                inflicted_damage = self._handle_ohko_move(target)
            case 'whole-field-effect':
                self._handle_whole_field_effect(move)
            case 'field-effect':
                self._handle_field_effect(move, target)
            case 'force-switch':
                raise NotImplementedError("Force-switch moves are not supported yet.")
            case 'unique':
                self._handle_unique_move(move, target)

        self._apply_move_effects(move, attacker, target, inflicted_damage, restored_health)

    def _can_execute_move(self, attacker: Pokemon, move: PokemonMove, target: Pokemon) -> bool:
        if attacker.is_incapacitated():
            self.state.log_event(f'{attacker.name} was unable to execute move')
            return False

        self.state.log_event(f'{attacker.name} used {move.name} on {target.name}')
        move.current_pp -= 1

        if self._is_protected(move, target):
            self.state.log_event(f'{target.name} protected itself')
            return False

        if not self.is_hit(move, attacker, target):
            self.state.log_event(f'{target.name} avoided the attack')
            return False

        return True

    def _handle_ailment_move(self, move: PokemonMove, target: Pokemon) -> None:
        if move.ailment_is_volatile():
            target.apply_volatile_status(cast(VolatileStatusCondition, move.ailment_type))
        elif target.non_volatile_status_condition is None:
            target.apply_non_volatile_status(cast(NonVolatileStatusCondition, move.ailment_type))

    def _handle_damage_move(self, move: PokemonMove, attacker: Pokemon, target: Pokemon) -> int:
        damage = self.calculate_damage(move, attacker, target)
        self.state.log_event(f'{target.name} took {damage} damage')
        return damage

    def _handle_damage_with_ailment_move(self, move: PokemonMove, attacker: Pokemon, target: Pokemon) -> int:
        damage = self._handle_damage_move(move, attacker, target)
        if random.randint(1, 100) <= move.ailment_chance:
            self._handle_ailment_move(move, target)
        return damage

    def _handle_swagger_move(self, move: PokemonMove, target: Pokemon) -> None:
        self._handle_ailment_move(move, target)
        self.boost_stat(move, target)

    def _handle_damage_with_stat_change(self, move: PokemonMove, attacker: Pokemon, target: Pokemon) -> int:
        damage = self._handle_damage_move(move, attacker, target)
        if random.randint(1, 100) <= move.stat_chance:
            affected_pokemon = attacker if move.category == 'damage+raise' else target
            self.boost_stat(move, affected_pokemon)
        return damage

    def _handle_damage_with_healing(self, move: PokemonMove, attacker: Pokemon, target: Pokemon) -> int:
        return self._handle_damage_move(move, attacker, target)

    def _handle_ohko_move(self, target: Pokemon) -> int:
        damage = target.current_hp
        self.state.log_event("It's a one-hit KO!")
        return damage

    def _handle_whole_field_effect(self, move: PokemonMove):
        if move.name == 'rain-dance':
            self.state.battle_effects_manager.set_weather('rain')
        elif move.name == 'sunny-day':
            self.state.battle_effects_manager.set_weather('sunshine')
        elif move.name in ['snowscape', 'chilly-reception']:
            self.state.battle_effects_manager.set_weather('snow')
        elif move.name == 'sandstorm':
            self.state.battle_effects_manager.set_weather('sandstorm')
        elif is_terrain(move.name):
            self.state.battle_effects_manager.set_terrain(move.name)
        elif move.name == 'haze':
            for pkm in self.state.battle_field:
                pkm.reset_boosts()


    def _handle_field_effect(self, move: PokemonMove, target: Pokemon):
        target_side = self.state.get_pokemon_side(target)

        if is_field(move.name):
            self.state.battle_effects_manager.add_field_effect(move.name, target_side)
        elif is_barrier(move.name):
            self.state.battle_effects_manager.add_barrier(move.name, target_side)
        elif is_hazard(move.name):
            self.state.battle_effects_manager.add_hazard(move.name, target_side)

    def _handle_unique_move(self, move, target):
        if move.name == 'protect':
            target.protected = True

    def _calculate_healing(self, move: PokemonMove, attacker: Pokemon) -> int:
        return math.floor(attacker.stats['hp'] * (move.healing / 100))

    def _is_protected(self, move: PokemonMove, target: Pokemon) -> bool:
        return (move.target in ['all-other-pokemon', 'random-opponent', 'selected-pokemon', 'all-opponents']
                and target.protected)

    def _apply_move_effects(self, move: PokemonMove, attacker: Pokemon, target: Pokemon,
                            inflicted_damage: int, restored_health: int) -> None:
        target.take_damage(inflicted_damage)
        attacker.restore_health(restored_health)

        # Handle flinching
        if random.randint(1, 100) <= move.flinch_chance and target.ability != 'inner-focus':
            target.flinched = True

        if target.is_fainted():
            self.state.log_event(f'{target.name} fainted')

        # Handle recoil
        if move.drain < 0:
            recoil_damage = int(inflicted_damage * (move.drain / 100))
            attacker.take_damage(recoil_damage)
            self.state.log_event(f'{attacker.name} was damaged by the recoil')

        if attacker.is_fainted():
            self.state.log_event(f'{attacker.name} fainted')

    def calculate_damage(self, move: PokemonMove, attacker: Pokemon, defender: Pokemon) -> int:
        """https://bulbapedia.bulbagarden.net/wiki/Damage#Generation_V_onward"""
        stab_modifier = 2 if move.type in attacker.types and attacker.ability == 'adaptability' else 1.5 if move.type in attacker.types else 1
        is_crit = self.is_critical_hit(move, attacker)
        if is_crit:
            self.state.log_event("It's a critical hit!")
        offensive_effective_stat, defensive_effective_stat = self.get_effective_stats(
            attacker=attacker, defender=defender, move=move, is_crit=is_crit)
        # We only model 1v1 battles, so this will always be 1
        targets_modifier = 1
        weather_modifier = self.get_weather_modifier(move)
        random_modifier = random.randint(85, 101) / 100
        type_modifier = self.get_type_effectiveness(move, defender)
        if type_modifier == 0:
            self.state.log_event("It didn't effect the opposing Pokemon")
        elif type_modifier < 1:
            self.state.log_event("It's not very effective")
        elif type_modifier > 1:
            self.state.log_event("It's super effective")
        burn_modifier = 0.5 if move.damage_class == 'physical' and attacker.non_volatile_status_condition and attacker.non_volatile_status_condition.name == 'burn' and attacker.ability != 'guts' and move.name != 'facade' else 1
        # Item, ability, aura boosts etc. (e.g., choice band)
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
            if self.state.battle_effects_manager.weather is not None and weather == self.state.battle_effects_manager.weather.name and affected_type in defender.types:
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
        for defender_type in defender.types:
            effectiveness *= TYPE_CHART.get(attacking_move.type, {}).get(defender_type, 1)
        return effectiveness

    def get_weather_modifier(self, move: PokemonMove) -> float:
        if self.state.battle_effects_manager.weather is None:
            return 1.0
        match self.state.battle_effects_manager.weather.name:
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

