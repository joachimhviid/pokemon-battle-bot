from pokemon_types import PokemonStatBoostStage


def get_stat_modifier(stat_stage: PokemonStatBoostStage) -> float:
    match stat_stage:
        case 0:
            return 2 / 2
        case 1:
            return 3 / 2
        case 2:
            return 4 / 2
        case 3:
            return 5 / 2
        case 4:
            return 6 / 2
        case 5:
            return 7 / 2
        case 6:
            return 8 / 2
        case -1:
            return 2 / 3
        case -2:
            return 2 / 4
        case -3:
            return 2 / 5
        case -4:
            return 2 / 6
        case -5:
            return 2 / 7
        case -6:
            return 2 / 8
        case _:
            return 1


if __name__ == "__main__":
    print('pokemon utils')
