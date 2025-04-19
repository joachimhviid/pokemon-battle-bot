from dataclasses import dataclass

from project.utils.constants import WeatherType


@dataclass
class Weather:
    name: WeatherType
    duration: int

    def encode(self) -> int:
        match self.name:
            case 'rain':
                return 1
            case 'sunshine':
                return 2
            case 'sandstorm':
                return 3
            case 'snow':
                return 4
        return 0