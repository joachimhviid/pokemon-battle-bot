from typing import Any, TypeAlias

import gymnasium as gym
import numpy as np

from project.battle.battle_actions import BattleActions
from project.battle.battle_state import BattleState
from project.core.pokemon import Pokemon, PokemonMove
from project.data.parsers import parse_team

# ObsType = dict[str, Union[np.integer, list[np.integer]]]
ObsType = dict[str, Any]
ActType: TypeAlias = int


class BattleEnv(gym.Env[ObsType, ActType]):
    def __init__(self, player_team: list[Pokemon], opponent_team: list[Pokemon]):
        MAX_PLAYER_SWITCH_OPTIONS = 6
        MAX_PLAYER_MOVES = 4
        MAX_MOVE_TARGETS = 2

        # TODO: implement masking and map integer to action
        self.action_space = gym.spaces.Discrete(MAX_PLAYER_MOVES * MAX_MOVE_TARGETS + MAX_PLAYER_SWITCH_OPTIONS)
        self.observation_space = gym.spaces.Dict({
            'player_active_pokemon': gym.spaces.Box(0, 1, shape=(11,), dtype=np.float32),
            'player_team': gym.spaces.Box(0, 1, shape=(6, 11), dtype=np.float32),
            'player_fields': gym.spaces.MultiBinary(5),
            'player_hazards': gym.spaces.MultiBinary(4),
            'player_barriers': gym.spaces.MultiBinary(3),

            'opponent_active_pokemon': gym.spaces.Box(0, 1, shape=(11,), dtype=np.float32),
            'opponent_team': gym.spaces.Box(0, 1, shape=(6, 11), dtype=np.float32),
            'opponent_fields': gym.spaces.MultiBinary(5),
            'opponent_hazards': gym.spaces.MultiBinary(4),
            'opponent_barriers': gym.spaces.MultiBinary(3),

            'weather': gym.spaces.Discrete(5),
            'terrain': gym.spaces.Discrete(5),
        })
        self.state = BattleState(player_team, opponent_team)
        self.actions = BattleActions(self.state)

    def action_to_move(self) -> dict[ActType, tuple[PokemonMove, Pokemon] | Pokemon]:
        """
        Should return a tuple of a move and a target or a switch to another Pokémon
        """
        return {
            0: (self.state.battle_field['player'][0].moves[0], self.state.battle_field['player'][0]),
            1: (self.state.battle_field['player'][0].moves[0], self.state.battle_field['opponent'][0]),
            2: (self.state.battle_field['player'][0].moves[1], self.state.battle_field['player'][0]),
            3: (self.state.battle_field['player'][0].moves[1], self.state.battle_field['opponent'][0]),
            4: (self.state.battle_field['player'][0].moves[2], self.state.battle_field['player'][0]),
            5: (self.state.battle_field['player'][0].moves[2], self.state.battle_field['opponent'][0]),
            6: (self.state.battle_field['player'][0].moves[3], self.state.battle_field['player'][0]),
            7: (self.state.battle_field['player'][0].moves[3], self.state.battle_field['opponent'][0]),
            8: self.state.player_team[0],
            9: self.state.player_team[1],
            10: self.state.player_team[2],
            11: self.state.player_team[3],
            12: self.state.player_team[4],
            13: self.state.player_team[5],
        }

    def step(self, action: ActType):  # type: ignore
        _action = self.action_to_move()[action]
        if isinstance(_action, tuple):
            move, _ = _action
            self.state.battle_field['player'][0].selected_move = move

        self.state.turn_counter += 1
        speed_sorted_pokemon = self.state.battle_field['player'] + self.state.battle_field['opponent']
        speed_sorted_pokemon.sort(key=lambda pkm: pkm.get_boosted_stat('speed'), reverse=True)

        self.on_turn_start(speed_sorted_pokemon)

        if isinstance(_action, tuple):
            move, target = _action
            for pkm in self.state.get_turn_order():
                if pkm.is_fainted():
                    continue
                print(f'{pkm.name} acts')
                self.actions.execute_move(move=move, attacker=pkm, target=target)
        else:
            for pkm in self.state.get_turn_order():
                print(f'{pkm.name} acts')
                self.actions.switch_pokemon(side=self.state.get_pokemon_side(pkm), selected_pokemon=_action)

        self.on_turn_end(speed_sorted_pokemon)

        terminated = self.state.get_winner() is not None
        truncated = False
        reward = 1 if self.state.get_winner() == 'player' else -1 if self.state.get_winner() == 'opponent' else 0
        observation = self.state.get_observation()
        self.validate_observation(observation)
        # info = self._get_info()
        return observation, reward, terminated, truncated, {}  # type: ignore

    def on_turn_start(self, sorted_active_pokemon: list[Pokemon]):
        for pkm in sorted_active_pokemon:
            pkm.on_turn_start()

    def on_turn_end(self, sorted_active_pokemon: list[Pokemon]):
        # trigger items (leftovers)
        for pkm in sorted_active_pokemon:
            pkm.on_turn_end()
        self.state.battle_effects_manager.on_turn_end(sorted_active_pokemon)

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        self.state.reset()
        observation = self.state.get_observation()
        self.validate_observation(observation)
        return observation, {}

    def validate_observation(self, observation: ObsType):
        assert self.observation_space.contains(observation), 'Invalid observation'


if __name__ == "__main__":
    team_1 = parse_team('player_1')
    team_2 = parse_team('player_2')
    env = BattleEnv(team_1, team_2)
    print(f'Team 1 first pokemon {env.battle_field["player"][0].name}')
    print(f'Team 2 first pokemon {env.battle_field["opponent"][0].name}')
    for i in range(10):
        env.execute_move(
            env.battle_field["player"][0].moves[3], env.battle_field["player"][0], env.battle_field["opponent"][0])
    for event in env.turn_events[0]:
        print(event)
