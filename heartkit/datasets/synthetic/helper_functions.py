import random
import warnings

import numpy as np
import numpy.typing as npt
from scipy.signal import savgol_filter

warnings.filterwarnings("ignore")


def evenly_spaced_y(original_x: npt.NDArray, y: npt.NDArray) -> npt.NDArray:
    """Evenly space y

    Args:
        original_x (npt.NDArray): Original x data
        y (npt.NDArray): Y data

    Returns:
        npt.NDArray: New y data
    """
    # Don't bother with the spacing function if snippet < 5 mS
    if y.shape[-1] < 5:
        return y

    # Transform a vector into an array so the function generalises to both
    if len(original_x.shape) == 1:
        original_x = np.expand_dims(original_x, axis=0)
        y = np.expand_dims(y, axis=0)

    new_x = np.arange(original_x.shape[1])
    intercepts = np.zeros((original_x.shape[0], new_x.shape[0]))

    for lead in range(original_x.shape[0]):
        np.seterr(divide="ignore")
        grads = (y[lead, :-1] - y[lead, 1:]) / (original_x[lead, :-1] - original_x[lead, 1:])
        placeholder = 0
        for i in range(new_x.shape[0]):
            for h in range(placeholder, original_x.shape[1], 1):
                if original_x[lead, h] >= new_x[i]:
                    intercepts[lead, i] = y[lead, h] + ((original_x[lead, h] - new_x[i]) * (-grads[max(h - 1, 0)]))
                    placeholder = h
                    break
            # END FOR
        # END FOR

    if intercepts.shape[0] == 1:
        intercepts = intercepts.reshape(intercepts.shape[1])

    return intercepts


def smooth_and_noise(
    y: npt.NDArray,
    rhythm: str = "sr",
    noise_multiplier: float = 1.0,
    impedance: float = 1.0,
) -> npt.NDArray:
    """Smooth and optionally add noise to signal

    Args:
        y (npt.NDArray): Signal data
        rhythm (str, optional): Rhythm preset. Defaults to "sr".
        noise_multiplier (float, optional): Global noise multiplier. Defaults to 1.0.
        impedance (float, optional): Performs y scaling. Defaults to 1.0.

    Returns:
        npt.NDArray: Signal data
    """
    y = y * (1 / impedance)

    # Generate baseline noise
    n = np.zeros((y.size,), dtype=complex)
    n[40:100] = np.exp(1j * np.random.uniform(0, np.pi * 2, (60,)))
    atrial_fibrillation_noise = np.fft.ifft(n)
    atrial_fibrillation_noise = savgol_filter(atrial_fibrillation_noise, 31, 2)
    atrial_fibrillation_noise = atrial_fibrillation_noise[: y.size] * random.uniform(0.01, 0.1)
    y = y + (atrial_fibrillation_noise * random.uniform(0, 1.3) * noise_multiplier)
    y = savgol_filter(y, 31, 2)

    # Generate random electrical noise from leads
    lead_noise = np.random.normal(0, 1 * 10**-4, y.size)  # IDDQD: 10**-5

    # Generate EMG frequency noise
    emg_noise = np.zeros(0)
    emg_noise_partial = np.sin(np.linspace(-0.5 * np.pi, 1.5 * np.pi, 1000) * 10000) * 10**-5
    for _ in range(y.size // 1000):
        emg_noise = np.concatenate((emg_noise, emg_noise_partial))
    emg_noise = np.concatenate((emg_noise, emg_noise_partial[: y.size % 1000]))

    # Combine lead and EMG noise, add to ECG
    noise = (emg_noise + lead_noise) * noise_multiplier

    # Randomly vary global amplitude
    y = (y + noise) * random.uniform(0.5, 3)

    # Add baseline wandering
    skew = np.linspace(0, random.uniform(0, 2) * np.pi, y.size)
    skew = np.sin(skew) * random.uniform(10**-3, 10**-4)
    y = y + skew

    return y
