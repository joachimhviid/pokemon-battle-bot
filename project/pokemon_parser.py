from pokemon import Pokemon
import json


def parse_team(file_name: str) -> list[Pokemon]:
    with open(f'project/input/{file_name}.json', 'r') as file_stream:
        data = json.loads(file_stream.read())
        team = []
        for pokemon in data:
            team.append(
                Pokemon(name=pokemon['name'], base_stats=pokemon['stats'], evs=pokemon['evs'], ivs=pokemon['ivs'],
                        nature=pokemon['nature'], level=pokemon['level'], moves=pokemon['moves'], types=pokemon['types'],
                        ability=pokemon['ability'], held_item=pokemon['held_item'])
            )
        return team


if __name__ == "__main__":
    team = parse_team('blastoise')
    print(team)
    parse_team('blastoise')
