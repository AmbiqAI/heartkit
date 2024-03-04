import numpy as np
import numpy.typing as npt
import physiokit as pk

from ..defines import AugmentationParams, PreprocessParams


def preprocess_pipeline(x: npt.NDArray, preprocesses: list[PreprocessParams], sample_rate: float) -> npt.NDArray:
    """Apply preprocessing pipeline

    Args:
        x (npt.NDArray): Signal
        preprocesses (list[PreprocessParams]): Preprocessing pipeline
        sample_rate (float): Sampling rate in Hz.

    Returns:
        npt.NDArray: Preprocessed signal
    """
    for preprocess in preprocesses:
        match preprocess.name:
            case "filter":
                x = pk.signal.filter_signal(x, sample_rate=sample_rate, **preprocess.params)
            case "znorm":
                x = pk.signal.normalize_signal(x, **preprocess.params)
            case _:
                raise ValueError(f"Unknown preprocess '{preprocess.name}'")
        # END MATCH
    # END FOR
    return x


def augment_pipeline(x: npt.NDArray, augmentations: list[AugmentationParams], sample_rate: float) -> npt.NDArray:
    """Apply augmentation pipeline

    Args:
        x (npt.NDArray): Signal
        augmentations (list[AugmentationParams]): Augmentations to apply
        sample_rate: Sampling rate in Hz.

    Returns:
        npt.NDArray: Augmented signal
    """
    x_sd = np.nanstd(x)
    for augmentation in augmentations:
        args = augmentation.params
        match augmentation.name:
            case "baseline_wander":
                amplitude = args.get("amplitude", [0.05, 0.06])
                frequency = args.get("frequency", [0, 1])
                x = pk.signal.add_baseline_wander(
                    x,
                    amplitude=np.random.uniform(amplitude[0], amplitude[1]),
                    frequency=np.random.uniform(frequency[0], frequency[1]),
                    sample_rate=sample_rate,
                    signal_sd=x_sd,
                )
            case "motion_noise":
                amplitude = args.get("amplitude", [0.5, 1.0])
                frequency = args.get("frequency", [0.4, 0.6])
                x = pk.signal.add_motion_noise(
                    x,
                    amplitude=np.random.uniform(amplitude[0], amplitude[1]),
                    frequency=np.random.uniform(frequency[0], frequency[1]),
                    sample_rate=sample_rate,
                    signal_sd=x_sd,
                )
            case "burst_noise":
                amplitude = args.get("amplitude", [0.05, 0.5])
                frequency = args.get("frequency", [sample_rate / 4, sample_rate / 2])
                burst_number = args.get("burst_number", [0, 2])
                x = pk.signal.add_burst_noise(
                    x,
                    amplitude=np.random.uniform(amplitude[0], amplitude[1]),
                    frequency=np.random.uniform(frequency[0], frequency[1]),
                    num_bursts=np.random.randint(burst_number[0], burst_number[1]),
                    sample_rate=sample_rate,
                    signal_sd=x_sd,
                )
            case "powerline_noise":
                amplitude = args.get("amplitude", [0.005, 0.01])
                frequency = args.get("frequency", [50, 60])
                x = pk.signal.add_powerline_noise(
                    x,
                    amplitude=np.random.uniform(amplitude[0], amplitude[1]),
                    frequency=np.random.uniform(frequency[0], frequency[1]),
                    sample_rate=sample_rate,
                    signal_sd=x_sd,
                )
            case "noise_sources":
                num_sources = args.get("num_sources", [1, 2])
                amplitude = args.get("amplitude", [0, 0.1])
                frequency = args.get("frequency", [0, sample_rate / 2])
                num_sources: int = np.random.randint(num_sources[0], num_sources[1])
                x = pk.signal.add_noise_sources(
                    x,
                    amplitudes=[np.random.uniform(amplitude[0], amplitude[1]) for _ in range(num_sources)],
                    frequencies=[np.random.uniform(frequency[0], frequency[1]) for _ in range(num_sources)],
                    noise_shapes=["laplace" for _ in range(num_sources)],
                    sample_rate=sample_rate,
                    signal_sd=x_sd,
                )
            case "lead_noise":
                scale = args.get("scale", [0.05, 0.25])
                x = pk.signal.add_lead_noise(
                    x,
                    scale=x_sd * np.random.uniform(scale[0], scale[1]),
                )
            case _:  # default
                pass
                # raise ValueError(f"Unknown augmentation '{augmentation.name}'")
        # END MATCH
    # END FOR
    return x
