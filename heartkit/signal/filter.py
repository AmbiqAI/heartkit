import functools

import numpy as np
import numpy.typing as npt
import scipy.signal as sps


@functools.cache
def get_butter_sos(
    lowcut: float | None = None,
    highcut: float | None = None,
    sample_rate: float = 1000,
    order: int = 2,
) -> npt.NDArray:
    """Compute biquad filter coefficients as SOS. This function caches.

    Args:
        lowcut (float|None): Lower cutoff in Hz. Defaults to None.
        highcut (float|None): Upper cutoff in Hz. Defaults to None.
        sample_rate (float): Sampling rate in Hz. Defaults to 1000 Hz.
        order (int, optional): Filter order. Defaults to 2.

    Returns:
        npt.NDArray: SOS
    """
    nyq = sample_rate / 2
    if lowcut is not None and highcut is not None:
        freqs = [lowcut / nyq, highcut / nyq]
        btype = "bandpass"
    elif lowcut is not None:
        freqs = lowcut / nyq
        btype = "highpass"
    elif highcut is not None:
        freqs = highcut / nyq
        btype = "lowpass"
    else:
        raise ValueError("At least one of lowcut or highcut must be specified")
    sos = sps.butter(order, freqs, btype=btype, output="sos")
    return sos


def resample_signal(
    data: npt.NDArray,
    sample_rate: float = 1000,
    target_rate: float = 500,
    axis: int = -1,
) -> npt.NDArray:
    """Resample signal using scipy FFT-based resample routine.

    Args:
        data (npt.NDArray): Signal
        sample_rate (float): Signal sampling rate. Defaults to 1000 Hz.
        target_rate (float): Target sampling rate. Defaults to 500 Hz.
        axis (int, optional): Axis to resample along. Defaults to -1.

    Returns:
        npt.NDArray: Resampled signal
    """
    if sample_rate == target_rate:
        return data
    desired_length = int(np.round(data.shape[axis] * target_rate / sample_rate))
    return sps.resample(data, desired_length, axis=axis)


def normalize_signal(data: npt.NDArray, eps: float = 1e-3, axis: int | None = None) -> npt.NDArray:
    """Normalize signal about its mean and std.

    Args:
        data (npt.NDArray): Signal
        eps (float, optional): Epsilon added to st. dev. Defaults to 1e-3.
        axis (int, optional): Axis to normalize along. Defaults to None.

    Returns:
        npt.NDArray: Normalized signal
    """
    mu = np.nanmean(data, axis=axis)
    std = np.nanstd(data, axis=axis) + eps
    return (data - mu) / std


def filter_signal(
    data: npt.NDArray,
    lowcut: float | None = None,
    highcut: float | None = None,
    sample_rate: float = 1000,
    order: int = 2,
    axis: int = -1,
    forward_backward: bool = True,
) -> npt.NDArray:
    """Apply SOS filter to signal using butterworth design and cascaded filter.

    Args:
        data (npt.NDArray): Signal
        lowcut (float|None): Lower cutoff in Hz. Defaults to None.
        highcut (float|None): Upper cutoff in Hz. Defaults to None.
        sample_rate (float): Sampling rate in Hz Defaults to 1000 Hz.
        order (int, optional): Filter order. Defaults to 2.
        forward_backward (bool, optional): Apply filter forward and backwards. Defaults to True.

    Returns:
        npt.NDArray: Filtered signal
    """
    sos = get_butter_sos(lowcut=lowcut, highcut=highcut, sample_rate=sample_rate, order=order)
    if forward_backward:
        return sps.sosfiltfilt(sos, data, axis=axis)
    return sps.sosfilt(sos, data, axis=axis)


def remove_baseline_wander(
    data: npt.NDArray,
    cutoff: float = 0.05,
    quality: float = 0.005,
    sample_rate: float = 1000,
    axis: int = -1,
    forward_backward: bool = True,
) -> npt.NDArray:
    """Remove baseline wander from signal using a notch filter.

    Args:
        data (npt.NDArray): Signal
        cutoff (float, optional): Cutoff frequency in Hz. Defaults to 0.05.
        quality (float, optional): Quality factor. Defaults to 0.005.
        sample_rate (float): Sampling rate in Hz. Defaults to 1000 Hz.
        axis (int, optional): Axis to filter along. Defaults to 0.
        forward_backward (bool, optional): Apply filter forward and backwards. Defaults to True.

    Returns:
        npt.NDArray: Filtered signal
    """
    b, a = sps.iirnotch(cutoff, Q=quality, fs=sample_rate)
    if forward_backward:
        return sps.filtfilt(b, a, data, axis=axis)
    return sps.lfilter(b, a, data, axis=axis)


def smooth_signal(
    data: npt.NDArray,
    window_length: int | None = None,
    polyorder: int = 3,
    sample_rate: float = 1000,
    axis: int = -1,
) -> npt.NDArray:
    """Smooths signal using savitzky-golay filter

    Args:
        data (npt.NDArray): Signal
        window_length (int | None, optional): Filter window length. Defaults to None.
        polyorder (int, optional): Poly fit order. Defaults to 3.
        sample_rate (float, optional): Sampling rate in Hz. Defaults to 1000 Hz.
        axis (int, optional): Axis to filter along. Defaults to -1.

    Returns:
        npt.NDArray: Smoothed signal
    """

    if window_length is None:
        window_length = sample_rate // 10

    if window_length % 2 == 0 or window_length == 0:
        window_length += 1

    return sps.savgol_filter(data, window_length=window_length, polyorder=polyorder, axis=axis)


def quotient_filter_mask(
    data: npt.NDArray,
    mask: npt.NDArray | None = None,
    iterations: int = 2,
    lowcut: float = 0.8,
    highcut: float = 1.2,
) -> npt.NDArray:
    """Applies a quotient filter to identify outliers from list.

    Args:
        data (npt.NDArray): Signal
        mask (npt.NDArray | None, optional): Rejection mask. Defaults to None.
        iterations (int, optional): # iterations to apply. Defaults to 2.
        lowcut (float, optional): Lower cutoff ratio. Defaults to 0.8.
        highcut (float, optional): Upper cutoff ratio. Defaults to 1.2.

    Returns:
        npt.NDArray: Rejection mask 0=accept, 1=reject.
    """

    if mask is None:
        mask = np.zeros_like(data, dtype=int)

    for _ in range(iterations):
        # Get indices of intervals to be filtered
        filt_idxs = np.where(mask == 0)[0]
        filt_ints = data[filt_idxs]
        # Compute quotient of each interval with the next
        filt_deltas = np.zeros(filt_ints.size)
        filt_deltas[1:] = filt_ints[:-1] / filt_ints[1:]
        filt_deltas[0] = filt_deltas[1]
        # Get indices of intervals that are outside the range
        delta_idxs = np.where((filt_deltas < lowcut) | (filt_deltas > highcut))[0]
        # Update mask with rejected intervals
        mask[filt_idxs[delta_idxs]] = 1
        # Break if no intervals are rejected
        if delta_idxs.size == 0:
            break
    # END FOR

    return mask
