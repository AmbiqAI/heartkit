import numpy.typing as npt
import physiokit as pk

from ..defines import PreprocessParams


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
