from enum import IntEnum


class HKSegment(IntEnum):
    """Heart segment labels"""

    normal = 0
    pwave = 1
    qrs = 2
    twave = 3
    uwave = 4  # Not used
    noise = 5
