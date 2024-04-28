from enum import IntEnum,auto

class Layer(IntEnum):
    BACKGROUND = auto()
    OBSTACLE = auto()
    GROUND = auto()
    PLAYER = auto()
    UI = auto()