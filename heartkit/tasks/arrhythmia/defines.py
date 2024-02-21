from enum import IntEnum, StrEnum


class HeartRhythm(IntEnum):
    """Heart rhythm labels"""

    normal = 0
    afib = 1  # Atrial fibrillation
    aflut = 2  # Atrial flutter
    sbrad = 3  # Sinus bradycardia
    stach = 4  # Sinus tachycardia
    sarrh = 5  # Sinus arrhythmia
    svarr = 6  # Supraventricular arrhythmia
    svt = 7  # Supraventricular tachycardia

    bigu = 8  # Bigeminy (every other beat is PVC)
    trigu = 9  # Trigeminy (every third beat is PVC)
    pace = 10  # Paced rhythm
    noise = 127  # Noise


class HeartRhythmName(StrEnum):
    """Heart rhythm label names"""

    normal = "normal"
    afib = "afib"
    aflut = "aflut"
    sbrad = "sbrad"
    stach = "stach"
    sarrh = "sarrh"
    svarr = "svarr"
    svt = "svt"
    bigu = "bigu"
    trigu = "trigu"
    pace = "pace"
    noise = "noise"


class HeartRate(IntEnum):
    """Heart rate labels"""

    normal = 0
    tachycardia = 1
    bradycardia = 2
    noise = 3  # Not used

    @classmethod
    def from_bpm(cls, bpm: float):
        """Assign rate based on supplied BPM."""
        if bpm < 60:
            return cls.bradycardia
        if bpm > 100:
            return cls.tachycardia
        return cls.normal


class HeartRateName(StrEnum):
    """Heart rate label names"""

    normal = "normal"
    tachycardia = "tachy"
    bradycardia = "brady"
    noise = "noise"
