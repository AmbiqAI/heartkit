import numpy.typing as npt
from scipy.signal import butter, sosfiltfilt


def butter_bp_filter(
    data: npt.ArrayLike,
    lowcut: float,
    highcut: float,
    sample_rate: float,
    order: int = 2,
) -> npt.ArrayLike:
    """Apply band-pass filter using butterworth  design and forward-backward cascaded filter

    Args:
        data (npt.ArrayLike): Data
        lowcut (float): Lower cutoff in Hz
        highcut (float): Upper cutoff in Hz
        sample_rate (float): Sampling rate in Hz
        order (int, optional): Filter order. Defaults to 2.

    Returns:
        npt.ArrayLike: Filtered data
    """
    nyq = 0.5 * sample_rate
    low = lowcut / nyq
    high = highcut / nyq
    sos = butter(order, [low, high], btype="band", output="sos")
    f_data = sosfiltfilt(sos, data, axis=0)
    return f_data
