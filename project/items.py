from enum_items import Item
from pokemon import Pokemon
from pokemon_parser import parse_team

import os

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
            if pokemon.stats["max_hp"] - damage == 0 or pokemon.stats["max_hp"] - damage ==  1:
                pokemon.stats["hp"] = 1
                pokemon.held_item = "None"
                print(f"Used {pokemon_item} with {Item.FOCUS_SASH}")

        case "Life Orb":
            pokemon.move[i].power = pokemon.move[i].power * 1.3
            pokemon.stats["hp"] -= pokemon.stats["max_hp"] * 0.1
            if pokemon.stats["hp"] <= 0:
                pokemon.fainted = True
            print(f"Matched {pokemon_item} with {Item.LIFE_ORB}")

        case "Rocky Helmet":
            if opponent.move[i].damage_type == "Physical":
                pokemon.opponent.stats["hp"]  -= pokemon.stats["hp"] * 0.0625
                print(f"Matched {pokemon_item} with {Item.ROCKY_HELMET}")

        case "Leftovers":
            print(f"Matched {pokemon_item} with {Item.LEFTOVERS}")
            pokemon.stats["hp"] += pokemon.stats["max_hp"] * 0.06

        case "Booster Energy":
            if pokemon.ability == "Protosynthesis" or pokemon.ability == "Quark-Drive" and environment.weater != "Harsh-Sunlight" or environment.terrain != "Electric-Terrain":
                pokemon.stats.sort()
                pokemon.stats[0] = pokemon.stats[0] * 1.5
                pokemon.held_item = "None"
                print(f"Matched {pokemon_item} with {Item.BOOSTER_ENERGY}")

        case "Covert Cloak":
            if pokemon.opponent == True and pokemon.opponent.move[i].secondary_effect == True
                pokemon.opponent.move[i].secondary_effect.ignore = True
                print(f"Matched {pokemon_item} with {Item.COVERT_CLOAK}")

        case "Choice Specs":
            if pokemon.moves[i].category == "Special":
                pokemon.stats["sp_atk"] = pokemon.stats["sp_atk"] * 1.5
                # set pokemon.move[i] to be usable only 
            print(f"Matched {pokemon_item} with {Item.CHOICE_SPECS}")

        case "Choice Scarf":
            pokemon.stats["speed"] = pokemon.stats["speed"] * 1.5
            # pokemon.moves[i] = "usable"
            print(f"Matched {pokemon_item} with {Item.CHOICE_SCARF}")

        case "Safety Goggles":

            if pokemon.oppennet == True & pokemon.moves == powders:
                pokemon.ignore_powder = True

            if environment.weather == "sandstorm" or environment.weather == "hail":
                pokemon.stats["hp"] = pokemon.stats["hp"] # ignore damage

            print(f"Matched {pokemon_item} with {Item.SAFETY_GOGGLES}")

        case "Lum Berry":
            if pokemon.status != "None":
                pokemon.status = "None"
                pokemon.held_item = "None"
                print(f"Used {pokemon_item} with {Item.LUM_BERRY}")


if __name__ == "__main__":
    team = parse_team('blastoise')
    match_item(team[0])
    print(team[0].held_item)

    