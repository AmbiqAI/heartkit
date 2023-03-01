import random

import numpy as np
import numpy.typing as npt
import scipy.signal

# Add baseline wonder
# Add random jitter / gaussian noise
# Add time warping? would need to apply for both


def apply_noise_to_signal(
    y: npt.NDArray,
    noise_multiplier: float = 1.0,
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
    # axis = 0

    # Generate baseline noise
    n = np.zeros((y.size,), dtype=complex)
    n[40:100] = np.exp(1j * np.random.uniform(0, np.pi * 2, (60,)))
    atrial_fibrillation_noise = np.fft.ifft(n)
    atrial_fibrillation_noise = scipy.signal.savgol_filter(atrial_fibrillation_noise, 31, 2)
    atrial_fibrillation_noise = atrial_fibrillation_noise[: y.size] * random.uniform(0.01, 0.1)
    y = y + (atrial_fibrillation_noise * random.uniform(0, 1.3) * noise_multiplier)
    y = scipy.signal.savgol_filter(y, 31, 2)

    # Generate random electrical noise from leads
    lead_noise = np.random.normal(0, 1 * 10**-5, y.size)

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
