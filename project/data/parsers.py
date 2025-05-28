from project.core.pokemon import Pokemon
import json
import os


def parse_team(file_name: str) -> list[Pokemon]:
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # path to `project/`
    file_path = os.path.join(base_dir, 'input', f'{file_name}.json')

    with open(file_path, 'r') as file_stream:
        data = json.loads(file_stream.read())
        team = []
        for pokemon in data:
            team.append(Pokemon(pokemon))
        return team


if __name__ == "__main__":
    team = parse_team('player_1')
    print(team)
