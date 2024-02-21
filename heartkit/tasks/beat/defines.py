from enum import IntEnum, StrEnum


class HeartBeat(IntEnum):
    """Heart beat labels"""

    normal = 0
    pac = 1
    pvc = 2
    noise = 3  # Not used


class HeartBeatName(StrEnum):
    """Heart beat label names"""

    normal = "normal"
    pac = "pac"
    pvc = "pvc"
    noise = "noise"
