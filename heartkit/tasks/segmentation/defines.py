from enum import IntEnum


class HKSegment(IntEnum):
    """Segmentation task labels"""

    normal = 0
    pwave = 1
    qrs = 2
    twave = 3
    uwave = 4  # Not used
    noise = 5
    systolic = 6
    diastolic = 7
