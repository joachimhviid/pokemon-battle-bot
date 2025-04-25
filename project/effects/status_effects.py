from dataclasses import dataclass

from project.utils.constants import VolatileStatusCondition, NonVolatileStatusCondition


@dataclass
class VolatileStatus:
    name: VolatileStatusCondition
    duration: int

    def encode(self) -> float:
        match self.name:
            case 'confusion':
                return 1.0 / 9
            case 'disable':
                return 2.0 / 9
            case 'encore':
                return 3.0 / 9
            case 'infatuation':
                return 4.0 / 9
            case 'ingrain':
                return 5.0 / 9
            case 'leech-seed':
                return 6.0 / 9
            case 'torment':
                return 7.0 / 9
            case 'trap':
                return 8.0 / 9
            case 'yawn':
                return 9.0 / 9
        return 0.0


@dataclass
class NonVolatileStatus:
    name: NonVolatileStatusCondition
    duration: int

    def encode(self) -> float:
        match self.name:
            case 'paralysis':
                return 1.0 / 6
            case 'sleep':
                return 2.0 / 6
            case 'burn':
                return 3.0 / 6
            case 'freeze':
                return 4.0 / 6
            case 'poison':
                return 5.0 / 6
            case 'bad-poison':
                return 6.0 / 6
        return 0.0
