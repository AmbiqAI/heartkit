from enum import IntEnum, StrEnum


class HKRhythm(IntEnum):
    """Rhythm task labels"""

    sr = 0  # Sinus rhythm
    sbrad = 1  # Sinus bradycardia
    stach = 2  # Sinus tachycardia

    sarrh = 3  # Sinus arrhythmia
    svarr = 4  # Supraventricular arrhythmia
    svt = 5  # Supraventricular tachycardia
    vtach = 6  # Ventricular tachycardia

    afib = 7  # Atrial fibrillation
    aflut = 8  # Atrial flutter

    vfib = 9  # Ventricular fibrillation
    vflut = 10  # Ventricular flutter

    bigu = 11  # Bigeminy (every other beat is PVC)
    trigu = 12  # Trigeminy (every third beat is PVC)
    pace = 13  # Paced rhythm
    noise = 127  # Noise


class HeartRate(IntEnum):
    """Rate task labels"""

    sinus = 0
    tachycardia = 1
    bradycardia = 2
    noise = 3

    @classmethod
    def from_bpm(cls, bpm: float):
        """Assign rate based on supplied BPM."""
        if bpm < 60:
            return cls.bradycardia
        if bpm > 100:
            return cls.tachycardia
        return cls.sinus


class HeartRateName(StrEnum):
    """Heart rate label names"""

    sinus = "sinus"
    tachycardia = "tachy"
    bradycardia = "brady"
