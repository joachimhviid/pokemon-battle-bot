from pokemon import Pokemon
import json


def parse_team(json_string: str) -> list[Pokemon]:
    data = json.loads(json_string)
    team = []
    for pokemon in data:
        team.append(
            Pokemon(name=pokemon['name'], base_stats=pokemon['stats'], evs=pokemon['evs'], ivs=pokemon['ivs'], 
                    nature=pokemon['nature'], level=pokemon['level'], moves=pokemon['moves'], types=pokemon['types'], 
                    ability=pokemon['ability'], held_item=pokemon['held_item'])
        )
    return team


if __name__ == "__main__":
    with open('input/player_1.json', 'r') as file_stream:
        parse_team(file_stream.read())
