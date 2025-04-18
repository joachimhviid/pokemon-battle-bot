from dataclasses import dataclass

from project.utils.constants import VolatileStatusCondition, NonVolatileStatusCondition


@dataclass
class VolatileStatus:
    name: VolatileStatusCondition
    duration: int

    def encode(self) -> float:
        match self.name:
            case 'confusion':
                return 1.0
            case 'disable':
                return 2.0
            case 'encore':
                return 3.0
            case 'infatuation':
                return 4.0
            case 'ingrain':
                return 5.0
            case 'leech-seed':
                return 6.0
            case 'torment':
                return 7.0
            case 'trap':
                return 8.0
            case 'yawn':
                return 9.0


@dataclass
class NonVolatileStatus:
    name: NonVolatileStatusCondition
    duration: int

    def encode(self) -> float:
        match self.name:
            case 'paralysis':
                return 1.0
            case 'sleep':
                return 2.0
            case 'burn':
                return 3.0
            case 'freeze':
                return 4.0
            case 'poison':
                return 5.0
            case 'bad-poison':
                return 6.0