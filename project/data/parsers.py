from project.core.pokemon import Pokemon
import json


def parse_team(file_name: str) -> list[Pokemon]:
    with open(f'../project/input/{file_name}.json', 'r') as file_stream:
        data = json.loads(file_stream.read())
        team = []
        for pokemon in data:
            team.append(Pokemon(pokemon))
        return team


if __name__ == "__main__":
    team = parse_team('player_1')
    print(team)
