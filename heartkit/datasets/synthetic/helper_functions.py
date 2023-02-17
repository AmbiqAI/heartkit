import random
import warnings

import numpy as np
from scipy.signal import savgol_filter

warnings.filterwarnings("ignore")


ecg_presets = [
    "LAHB",
    "LPHB",
    "high_take_off",
    "LBBB",
    "ant_STEMI",
    "random_morphology",
]

segmentation_key = {
    0: "background",
    1: "P wave",
    # 16 : "Biphasic P wave",
    2: "PR interval",
    3: "QRS complex",
    # 17 : "Broad QRS with RSR pattern",
    # 18 : "Broad QRS without RSR pattern",
    # 19 : "Inverted narrow QRS complex",
    # 20 : "Inverted QRS with RSR pattern",
    # 21 : "Inverted QRS without RSR pattern",
    4: "ST segment",
    # 22 : "Upsloping ST segment",
    # 23 : "Downsloping ST segment",
    5: "T wave",
    # 24 : "Inverted T wave",
    6: "T-P segment",
    7: "T/P overlap",
    8: "start of P",
    9: "start of Q wave",
    10: "Q trough",
    11: "R peak",
    12: "R prime peak",
    13: "S trough",
    14: "J point",
    15: "end of QT segment (max slope method)",
    16: "end of T wave (absolute method)",
}

# the classes that represent ECG points rather than waves:
# (note that point 25 - T wave end - is included in waves as it is used to
# create bounding boxes)
points = [8, 9, 10, 11, 12, 13, 14, 15]
# points that should be present for every beat:
vital_points = [9, 11, 14]
# maps points to waves:
point_mapper = {
    8: "P wave",
    9: "QRS",
    10: "QRS",
    11: "QRS",
    12: "QRS",
    13: "QRS",
    14: "QRS",
    15: 5,
}
# the classes that represent ECG waves:
waves = np.setdiff1d([*range(len(segmentation_key))], points).tolist()

n_classes = len(waves) - 1


def evenly_spaced_y(original_x, y):
    # Don't bother with the spacing function if snippet < 5 mS
    if y.shape[-1] < 5:
        return y

    else:
        # Transform a vector into an array so the function generalises to both:
        if len(original_x.shape) == 1:
            original_x = np.expand_dims(original_x, axis=0)
            y = np.expand_dims(y, axis=0)

        new_x = np.arange(original_x.shape[1])
        intercepts = np.zeros((original_x.shape[0], new_x.shape[0]))

        for lead in range(original_x.shape[0]):
            np.seterr(divide="ignore")
            grads = (y[lead, :-1] - y[lead, 1:]) / (
                original_x[lead, :-1] - original_x[lead, 1:]
            )
            placeholder = 0
            for i in range(new_x.shape[0]):
                for h in range(placeholder, original_x.shape[1], 1):
                    if original_x[lead, h] >= new_x[i]:
                        intercepts[lead, i] = y[lead, h] + (
                            (original_x[lead, h] - new_x[i]) * (-grads[max(h - 1, 0)])
                        )
                        placeholder = h
                        break

        if intercepts.shape[0] == 1:
            intercepts = intercepts.reshape(intercepts.shape[1])

        return intercepts


def smooth_and_noise(y, rhythm="sr", universal_noise_multiplier=1.0, impedance=1.0):
    # universal_noise_multiplier is a single 'volume control' for all noise types
    y = y * (1 / impedance)

    # generate baseline noise:
    n = np.zeros((y.size,), dtype=complex)
    n[40:100] = np.exp(1j * np.random.uniform(0, np.pi * 2, (60,)))
    atrial_fibrillation_noise = np.fft.ifft(n)
    atrial_fibrillation_noise = savgol_filter(atrial_fibrillation_noise, 31, 2)
    atrial_fibrillation_noise = atrial_fibrillation_noise[: y.size] * random.uniform(
        0.01, 0.1
    )
    y = y + (
        atrial_fibrillation_noise * random.uniform(0, 1.3) * universal_noise_multiplier
    )
    y = savgol_filter(y, 31, 2)

    # generate random electrical noise from leads
    lead_noise = np.random.normal(0, 1 * 10**-5, y.size)

    # generate EMG frequency noise
    emg_noise = np.zeros(0)
    emg_noise_partial = (
        np.sin(np.linspace(-0.5 * np.pi, 1.5 * np.pi, 1000) * 10000) * 10**-5
    )
    for r in range(y.size // 1000):
        emg_noise = np.concatenate((emg_noise, emg_noise_partial))
    emg_noise = np.concatenate((emg_noise, emg_noise_partial[: y.size % 1000]))

    # combine lead and EMG noise, add to ECG
    noise = (emg_noise + lead_noise) * universal_noise_multiplier

    # randomly vary global amplitude
    y = (y + noise) * random.uniform(0.5, 3)

    # add baseline wandering
    skew = np.linspace(0, random.uniform(0, 2) * np.pi, y.size)
    skew = np.sin(skew) * random.uniform(10**-3, 10**-4)
    y = y + skew

    return y
