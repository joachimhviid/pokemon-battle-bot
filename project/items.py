from enum_items import Item
from pokemon import Pokemon
from pokemon_parser import parse_team


def map_items():
    items_array = []
    for i in Item:
        items_array.append(i)
    return items_array

# TODO Implement a function that matches a pokemon item with its corresponding item in the enum

def match_item(pokemon: Pokemon):
    # TODO implement each items mechanics
    pokemon_item = pokemon.held_item
    match pokemon_item:
        case "Sitrus Berry":
            if pokemon.stats["hp"] <= pokemon.stats["hp"] * 0.5:
                pokemon.stats["hp"] += pokemon.stats["hp"] * 0.25
                pokemon.held_item = "None" 
            print(f"Used {pokemon_item} with {Item.SITRUS_BERRY}")
        case "Focus Sash":
#            if pokemon.stats["max_hp"] - damage == 0 or pokemon.stats["max_hp"] - damage ==  1:
#                pokemon.stats["hp"] = 1
#                pokemon.held_item = "None"
                print(f"Used {pokemon_item} with {Item.FOCUS_SASH}")
        case "Life Orb":
            print(f"Matched {pokemon_item} with {Item.LIFE_ORB}")
        case "Rocky Helmet":
            print(f"Matched {pokemon_item} with {Item.ROCKY_HELMET}")
        case "Leftovers":
            print(f"Matched {pokemon_item} with {Item.LEFTOVERS}")
        case "Booster Energy":
            print(f"Matched {pokemon_item} with {Item.BOOSTER_ENERGY}")
        case "Covert Cloak":
            print(f"Matched {pokemon_item} with {Item.COVERT_CLOAK}")
        case "Choice Specs":
            print(f"Matched {pokemon_item} with {Item.CHOICE_SPECS}")
        case "Choice Scarf":
            print(f"Matched {pokemon_item} with {Item.CHOICE_SCARF}")
        case "Safety Goggles":
            print(f"Matched {pokemon_item} with {Item.SAFETY_GOGGLES}")
        case "Lum Berry":
            print(f"Matched {pokemon_item} with {Item.LUM_BERRY}")


if __name__ == "__main__":
    team = parse_team('blastoise')
    match_item(team[0])

    