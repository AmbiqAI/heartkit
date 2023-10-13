import numpy as np
import numpy.typing as npt
import physiokit as pk

from ..defines import AugmentationParams


def augment_pipeline(x: npt.NDArray, augmentations: list[AugmentationParams], sample_rate: float) -> npt.NDArray:
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
            x = pk.signal.add_baseline_wander(
                x,
                amplitude=np.random.uniform(amplitude[0], amplitude[1]),
                frequency=np.random.uniform(frequency[0], frequency[1]),
                sample_rate=sample_rate,
            )
        elif augmentation.name == "motion_noise":
            amplitude = args.get("amplitude", [0.5, 1.0])
            frequency = args.get("frequency", [0.4, 0.6])
            x = pk.signal.add_motion_noise(
                x,
                amplitude=np.random.uniform(amplitude[0], amplitude[1]),
                frequency=np.random.uniform(frequency[0], frequency[1]),
                sample_rate=sample_rate,
            )
        elif augmentation.name == "burst_noise":
            amplitude = args.get("amplitude", [0.05, 0.5])
            frequency = args.get("frequency", [sample_rate / 4, sample_rate / 2])
            burst_number = args.get("burst_number", [0, 2])
            x = pk.signal.add_burst_noise(
                x,
                amplitude=np.random.uniform(amplitude[0], amplitude[1]),
                frequency=np.random.uniform(frequency[0], frequency[1]),
                burst_number=np.random.randint(burst_number[0], burst_number[1]),
                sample_rate=sample_rate,
            )
        elif augmentation.name == "powerline_noise":
            amplitude = args.get("amplitude", [0.005, 0.01])
            frequency = args.get("frequency", [50, 60])
            x = pk.signal.add_powerline_noise(
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
            x = pk.signal.add_noise_sources(
                x,
                amplitudes=[np.random.uniform(amplitude[0], amplitude[1]) for _ in range(num_sources)],
                frequencies=[np.random.uniform(frequency[0], frequency[1]) for _ in range(num_sources)],
                sample_rate=sample_rate,
            )
        # END IF
    # END FOR
    return x
