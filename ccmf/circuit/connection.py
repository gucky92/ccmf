from enum import Enum


class Sign(Enum):
    EXCITATORY = 1
    INHIBITORY = 2
    UNSPECIFIED = 3


class Connection:
    def __init__(self, sign):
        self._sign = sign
