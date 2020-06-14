from enum import Enum


class Sign(Enum):
    EXCITATORY = 1
    INHIBITORY = 2
    UNSPECIFIED = 3

    def __str__(self):
        return self.name.capitalize()
