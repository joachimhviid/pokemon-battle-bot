from typing import Any, TypeAlias, List, Optional, cast

import gymnasium as gym
import numpy as np

from project import BattleAgent
from project.battle.battle_actions import BattleActions
from project.battle.battle_state import BattleState
from project.core.pokemon import Pokemon, PokemonMove
from project.data.parsers import parse_team
from project.utils.constants import Side

# ObsType = dict[str, Union[np.integer, list[np.integer]]]
ObsType = dict[str, Any]
ActType: TypeAlias = int


class BattleEnv(gym.Env[ObsType, ActType]):
    player_agent: Optional[BattleAgent]
    opponent_agent: Optional[BattleAgent]

    def __init__(self, player_team: list[Pokemon], opponent_team: list[Pokemon]):
        MAX_PLAYER_SWITCH_OPTIONS = 6
        MAX_PLAYER_MOVES = 4
        MAX_MOVE_TARGETS = 2

        self.action_space_size = MAX_PLAYER_MOVES * MAX_MOVE_TARGETS + MAX_PLAYER_SWITCH_OPTIONS
        self.action_space = gym.spaces.Discrete(self.action_space_size)
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
        self.player_agent = None
        self.opponent_agent = None

    def action_to_move(self, side: Side) -> dict[ActType, tuple[PokemonMove, int] | int]:
        """
        Should return a tuple of a move and a target or a switch to another Pokémon
        """
        active_pokemon = self.state.battle_field[0] if side == 'player' else self.state.battle_field[1]
        return {
            0: (active_pokemon.moves[0], 0),  # player and target index
            1: (active_pokemon.moves[0], 1),
            2: (active_pokemon.moves[1], 0),
            3: (active_pokemon.moves[1], 1),
            4: (active_pokemon.moves[2], 0),
            5: (active_pokemon.moves[2], 1),
            6: (active_pokemon.moves[3], 0),
            7: (active_pokemon.moves[3], 1),
            8: 0,  # team member indexes
            9: 1,
            10: 2,
            11: 3,
            12: 4,
            13: 5,
        }

    def get_action_mask(self, side: Side) -> np.ndarray:
        # has_valid_action = False

        active_pokemon = self.state.battle_field[0] if side == 'player' else self.state.battle_field[1]
        team = self.state.player_team if side == 'player' else self.state.opponent_team

        mask = np.zeros(self.action_space_size, dtype=np.bool)

        if active_pokemon.is_fainted():
            for i, pkm in enumerate(team):
                switch_index = 8 + i  # 8 is the beginning of team switch options
                if not pkm.is_fainted() and not pkm.active:
                    mask[switch_index] = True
            return mask

        for i, move in enumerate(active_pokemon.moves):
            # TODO: Only allow valid targets
            if move.current_pp > 0:  # Only allow moves with PP remaining
                mask[i * 2] = True  # Target 0
                mask[i * 2 + 1] = True  # Target 1

        for i, pkm in enumerate(team):
            switch_index = 8 + i  # 8 is the beginning of team switch options
            if not pkm.is_fainted() and not pkm.active:
                mask[switch_index] = True

        return mask

    def handle_action(self, action: tuple[PokemonMove, int] | int, side: Side):
        active_pokemon = self.state.battle_field[0] if side == 'player' else self.state.battle_field[1]
        if isinstance(action, tuple):
            move, target_index = action
            active_pokemon.selected_move = move

    def step(self, action: ActType):  # type: ignore
        # Player action
        player_action = self.action_to_move('player')[action]
        opponent_action = None
        if self.opponent_agent:
            opponent_observation = self.state.get_observation()
            opponent_action_mask = self.get_action_mask('opponent')
            opponent_action_index = self.opponent_agent.choose_action(opponent_observation, opponent_action_mask)
            opponent_action = self.action_to_move('opponent')[opponent_action_index]

        # print(player_action, opponent_action)

        # Set players selected move
        if isinstance(player_action, tuple):
            move, _ = player_action
            self.state.battle_field[0].selected_move = move

        # Set opponents selected move
        if opponent_action and isinstance(opponent_action, tuple):
            move, _ = opponent_action
            self.state.battle_field[1].selected_move = move

        self.state.turn_counter += 1
        speed_sorted_pokemon: List[Pokemon] = list(self.state.battle_field)
        speed_sorted_pokemon.sort(key=lambda pkm: pkm.get_boosted_stat('speed'), reverse=True)

        self.on_turn_start(speed_sorted_pokemon)

        for side, action in [('player', player_action), ('opponent', opponent_action)]:
            if action is not None and not isinstance(action, tuple):
                selected_pokemon = (self.state.player_team[action]
                                    if side == 'player'
                                    else self.state.opponent_team[action])
                self.actions.switch_pokemon(side=side, selected_pokemon=selected_pokemon)

        for pkm in self.state.get_turn_order():
            if pkm.is_fainted():
                continue

            current_side = self.state.get_pokemon_side(pkm)
            current_action = player_action if current_side == 'player' else opponent_action

            if current_action is None or not isinstance(current_action, tuple):
                continue

            move, target_index = current_action
            target = self.state.battle_field[target_index]
            self.actions.execute_move(move=move, attacker=pkm, target=target)

        for side, active_pokemon in [('player', self.state.battle_field[0]),
                                     ('opponent', self.state.battle_field[1])]:
            _side = cast(Side, side)
            if active_pokemon.is_fainted():
                # Get a new observation and mask for the switch
                observation = self.state.get_observation()
                action_mask = self.get_action_mask(_side)

                # Check if there are any valid switches available
                if any(action_mask[8:14]):  # Indices 8-13 represent switches
                    # Force the appropriate agent to choose a switch
                    if side == 'player':
                        switch_action = self.player_agent.choose_action(observation, action_mask)
                        pokemon_index = self.action_to_move(_side)[switch_action]
                        selected_pokemon = self.state.player_team[pokemon_index]
                    else:
                        switch_action = self.opponent_agent.choose_action(observation, action_mask)
                        pokemon_index = self.action_to_move(_side)[switch_action]
                        selected_pokemon = self.state.opponent_team[pokemon_index]

                    self.actions.switch_pokemon(side=_side, selected_pokemon=selected_pokemon)

        self.on_turn_end(speed_sorted_pokemon)

        terminated = self.state.get_winner() is not None
        truncated = False
        reward = 1 if self.state.get_winner() == 'player' else -1 if self.state.get_winner() == 'opponent' else 0
        observation = self.state.get_observation()
        self.validate_observation(observation)
        # info = self._get_info()
        if terminated:
            self.state.log_event(f'{str(self.state.get_winner())} wins!')
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
        assert self.observation_space.contains(observation), f'Invalid observation {observation}'


if __name__ == "__main__":
    team_1 = parse_team('player_1')
    team_2 = parse_team('player_2')
    env = BattleEnv(team_1, team_2)
    print(f'Team 1 first pokemon {env.state.battle_field[0].name}')
    print(f'Team 2 first pokemon {env.state.battle_field[1].name}')
    for i in range(10):
        env.actions.execute_move(
            env.state.battle_field[0].moves[3], env.state.battle_field[0], env.state.battle_field[1])
    for event in env.state.turn_events[0]:
        print(event)
