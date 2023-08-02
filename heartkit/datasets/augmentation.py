import random

import numpy as np
import numpy.typing as npt

from .. import signal
from ..defines import AugmentationParams


def emg_noise(
    y: npt.NDArray, scale: float = 1e-5, sampling_frequency: int = 1000
) -> npt.NDArray:
    """Add EMG noise to signal

    Args:
        y (npt.NDArray): Signal
        scale (float, optional): Noise scale. Defaults to 1e-5.
        sampling_frequency (int, optional): Sampling rate in Hz. Defaults to 1000.

    Returns:
        npt.NDArray: New signal
    """
    noise = np.repeat(
        np.sin(np.linspace(-0.5 * np.pi, 1.5 * np.pi, sampling_frequency) * 10000),
        np.ceil(y.size / sampling_frequency),
    )
    return y + scale * noise[: y.size]


def lead_noise(y: npt.NDArray, scale: float = 1) -> npt.NDArray:
    """Add Lead noise

    Args:
        y (npt.NDArray): Signal
        scale (float, optional): Noise scale. Defaults to 1.

    Returns:
        npt.NDArray: New signal
    """
    return y + np.random.normal(-scale, scale, size=y.shape)


def random_scaling(
    y: npt.NDArray, lower: float = 0.5, upper: float = 2.0
) -> npt.NDArray:
    """Randomly scale signal.

    Args:
        y (npt.NDArray): Signal
        lower (float, optional): Lower bound. Defaults to 0.5.
        upper (float, optional): Upper bound. Defaults to 2.0.

    Returns:
        npt.NDArray: New signal
    """
    return y * random.uniform(lower, upper)


def baseline_wander(y: npt.NDArray, scale: float = 1e-3) -> npt.NDArray:
    """Apply baseline wander

    Args:
        y (npt.NDArray): Signal
        scale (float, optional): Noise scale. Defaults to 1e-3.

    Returns:
        npt.NDArray: New signal
    """
    skew = np.linspace(0, random.uniform(0, 2) * np.pi, y.size)
    skew = np.sin(skew) * random.uniform(scale / 10, scale)
    y = y + skew


def augment_pipeline(
    x: npt.NDArray, augmentations: list[AugmentationParams], sample_rate: float
) -> npt.NDArray:
    """Apply augmentation pipeline
    Args:
        x (npt.NDArray): Signal
        augmentations (list[AugmentationParams]): Augmentations to apply
        sample_rate: Sampling rate in Hz.
    Returns:
        npt.NDArray: Augmented signal
    """
    for augmentation in augmentations:
        args = augmentation.args
        if augmentation.name == "baseline_wander":
            amplitude = args.get("amplitude", [0.05, 0.06])
            frequency = args.get("frequency", [0, 1])
            x = signal.add_baseline_wander(
                x,
                amplitude=np.random.uniform(amplitude[0], amplitude[1]),
                frequency=np.random.uniform(frequency[0], frequency[1]),
                sample_rate=sample_rate,
            )
        elif augmentation.name == "motion_noise":
            amplitude = args.get("amplitude", [0.5, 1.0])
            frequency = args.get("frequency", [0.4, 0.6])
            x = signal.add_motion_noise(
                x,
                amplitude=np.random.uniform(amplitude[0], amplitude[1]),
                frequency=np.random.uniform(frequency[0], frequency[1]),
                sample_rate=sample_rate,
            )
        elif augmentation.name == "burst_noise":
            amplitude = args.get("amplitude", [0.05, 0.5])
            frequency = args.get("frequency", [sample_rate / 4, sample_rate / 2])
            burst_number = args.get("burst_number", [0, 2])
            x = signal.add_burst_noise(
                x,
                amplitude=np.random.uniform(amplitude[0], amplitude[1]),
                frequency=np.random.uniform(frequency[0], frequency[1]),
                burst_number=np.random.randint(burst_number[0], burst_number[1]),
                sample_rate=sample_rate,
            )
        elif augmentation.name == "powerline_noise":
            amplitude = args.get("amplitude", [0.005, 0.01])
            frequency = args.get("frequency", [50, 60])
            x = signal.add_powerline_noise(
                x,
                amplitude=np.random.uniform(amplitude[0], amplitude[1]),
                frequency=np.random.uniform(frequency[0], frequency[1]),
                sample_rate=sample_rate,
            )
        elif augmentation.name == "noise_sources":
            num_sources = args.get("num_sources", [1, 2])
            amplitude = args.get("amplitude", [0, 0.1])
            frequency = args.get("frequency", [0, sample_rate / 2])
            num_sources: int = np.random.randint(num_sources[0], num_sources[1])
            x = signal.add_noise_sources(
                x,
                amplitudes=[
                    np.random.uniform(amplitude[0], amplitude[1])
                    for _ in range(num_sources)
                ],
                frequencies=[
                    np.random.uniform(frequency[0], frequency[1])
                    for _ in range(num_sources)
                ],
                sample_rate=sample_rate,
            )
    return x
