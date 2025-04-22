import random
from typing import Any, List, Dict, Tuple

import numpy as np

from project.core.pokemon import Pokemon
from project.effects.effects_manager import BattleEffectsManager
from project.utils.constants import Side


class BattleState:
    def __init__(self, player_team: List[Pokemon], opponent_team: List[Pokemon]):
        self.turn_counter: int = 0
        self.turn_events: Dict[int, list[str]] = {}
        self.battle_effects_manager: BattleEffectsManager = BattleEffectsManager()

        self.player_team: List[Pokemon] = player_team
        self.opponent_team: List[Pokemon] = opponent_team
        self.battle_field: List[Pokemon] = [player_team[0], opponent_team[0]]

        for pkm in self.battle_field:
            pkm.on_switch_in()

    def encode_team(self, team: List[Pokemon]) -> np.ndarray[Any, np.dtype[np.float32]]:
        team_vecs = [pkm.encode() for pkm in team]
        # Pad with zeros if fewer than 6 Pokémon
        while len(team_vecs) < 6:
            team_vecs.append(np.zeros(11, dtype=np.float32))
        return np.clip(np.stack(team_vecs[:6]), 0.0, 1.0)

    def get_observation(self) -> Dict[str, Any]:
        return {
            'player_active_pokemon': self.battle_field[0].encode(),
            'player_team': self.encode_team(self.player_team),
            'player_fields': self.battle_effects_manager.encode_fields('player'),
            'player_barriers': self.battle_effects_manager.encode_barriers('player'),
            'player_hazards': self.battle_effects_manager.encode_hazards('player'),

            'opponent_active_pokemon': self.battle_field[1].encode(),
            'opponent_team': self.encode_team(self.opponent_team),
            'opponent_fields': self.battle_effects_manager.encode_fields('opponent'),
            'opponent_barriers': self.battle_effects_manager.encode_barriers('opponent'),
            'opponent_hazards': self.battle_effects_manager.encode_hazards('opponent'),

            'weather': self.battle_effects_manager.encode_weather(),
            'terrain': self.battle_effects_manager.encode_terrain(),
        }

    def get_pokemon_side(self, pokemon: Pokemon) -> Side:
        if pokemon in self.player_team:
            return 'player'
        elif pokemon in self.opponent_team:
            return 'opponent'
        else:
            raise ValueError("The given Pokemon does not belong to either team.")

    def get_turn_order(self) -> List[Pokemon]:
        active_pokemon: List[Pokemon] = list(self.battle_field)
        active_pokemon.sort(
            key=lambda pkm: (pkm.selected_move.priority, pkm.get_boosted_stat('speed')),
            reverse=True
        )
        if (active_pokemon[0].stats['speed'] == active_pokemon[1].stats['speed'] and
                active_pokemon[0].selected_move.priority == active_pokemon[1].selected_move.priority):
            random.shuffle(active_pokemon)
        return active_pokemon

    def get_winner(self) -> Side | None:
        player_fainted = all(pokemon.is_fainted() for pokemon in self.player_team)
        opponent_fainted = all(pokemon.is_fainted() for pokemon in self.opponent_team)
        return 'player' if opponent_fainted else 'opponent' if player_fainted else None

    def log_event(self, battle_event: str):
        if self.turn_counter not in self.turn_events:
            self.turn_events[self.turn_counter] = []
        self.turn_events[self.turn_counter].append(battle_event)

    def print_turn_events(self, file_path: str | None = None):
        for turn, events in self.turn_events.items():
            if file_path:
                with open(file_path, 'a', encoding='utf-8') as f:
                    f.write(f"Turn {turn}:\n")
                    for event in events:
                        f.write(f"- {event}\n")
                    f.write("\n")
            else:
                print(f"Turn {turn}:")
                for event in events:
                    print(f"- {event}")

    def reset(self):
        self.turn_counter = 0
        self.turn_events.clear()
        for pokemon in self.player_team:
            pokemon.reset()
        for pokemon in self.opponent_team:
            pokemon.reset()
        self.battle_effects_manager.reset()
