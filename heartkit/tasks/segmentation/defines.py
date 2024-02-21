from enum import IntEnum, StrEnum


class HeartSegment(IntEnum):
    """Heart segment labels"""

    normal = 0
    pwave = 1
    qrs = 2
    twave = 3
    uwave = 4  # Not used
    noise = 5


class HeartSegmentName(StrEnum):
    """Heart segment names"""

    normal = "normal"
    pwave = "pwave"
    qrs = "qrs"
    twave = "twave"
    uwave = "uwave"  # Not used
