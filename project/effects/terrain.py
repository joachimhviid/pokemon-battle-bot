from dataclasses import dataclass

from project.utils.constants import TerrainType


@dataclass
class Terrain:
    name: TerrainType
    duration: int

    def encode(self) -> int:
        match self.name:
            case 'electric-terrain':
                return 1
            case 'grassy-terrain':
                return 2
            case 'misty-terrain':
                return 3
            case 'psychic-terrain':
                return 4
        return 0