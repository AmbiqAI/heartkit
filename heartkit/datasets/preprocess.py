import functools
import os
from typing import Literal

import numpy as np
import numpy.typing as npt
import scipy.signal


def preprocess_signal(
    data: npt.ArrayLike, sample_rate: float, target_rate: float | None = None
) -> npt.NDArray:
    """Pre-process signal

    Args:
        data (npt.ArrayLike): Signal
        sample_rate (float): Sampling rate (Hz)
        target_rate (float): Target sampling rate (Hz)

    Returns:
        npt.ArrayLike: Pre-processed signal
    """
    axis = 0
    norm_en = True
    norm_eps = 1e-6
    filter_en = True
    filt_lo = 0.5
    filt_hi = 30
    resample_en = target_rate is not None and sample_rate != target_rate

    x = np.copy(data)

    if filter_en:
        x = filter_signal(
            x, lowcut=filt_lo, highcut=filt_hi, sample_rate=sample_rate, axis=axis
        )
    if resample_en:
        x = resample_signal(
            x, sample_rate=sample_rate, target_rate=target_rate, axis=axis
        )
    if norm_en:
        x = normalize_signal(x, eps=norm_eps)
    return x


@functools.cache
def get_butter_bp_sos(
    lowcut: float,
    highcut: float,
    sample_rate: float,
    order: int = 3,
) -> npt.ArrayLike:
    """Compute band-pass filter coefficients as SOS. This function caches.

    Args:
        lowcut (float): Lower cutoff in Hz
        highcut (float): Upper cutoff in Hz
        sample_rate (float): Sampling rate in Hz
        order (int, optional): Filter order. Defaults to 3.
    Returns:
        npt.ArrayLike: SOS
    """
    nyq = 0.5 * sample_rate
    low = lowcut / nyq
    high = highcut / nyq
    sos = scipy.signal.butter(order, [low, high], btype="band", output="sos")
    return sos


def generate_arm_biquad_sos(
    lowcut: float,
    highcut: float,
    sample_rate: float,
    order: int = 3,
    var_name: str = "biquadFilter",
):
    """Generate ARM second order section coefficients."""
    sos = get_butter_bp_sos(
        lowcut=lowcut, highcut=highcut, sample_rate=sample_rate, order=order
    )
    # Each section needs to be mapped as follows:
    #   [b0, b1, b2, a0, a1, a2] -> [b0, b1, b2, -a1, -a2]
    sec_len_name = f"{var_name.upper()}_NUM_SECS"
    arm_sos = sos[:, [0, 1, 2, 4, 5]] * [1, 1, 1, -1, -1]
    coef_str = ", ".join(
        f"{os.linesep:<4}{c}" if i % 5 == 0 else f"{c}"
        for i, c in enumerate(arm_sos.flatten())
    )
    arm_str = (
        f"#define {sec_len_name} ({order:0d}){os.linesep}"
        f"static float32_t {var_name}State[2 * {sec_len_name}];{os.linesep}"
        f"static float32_t {var_name}[5 * {sec_len_name}] = {{ {coef_str}\n}};{os.linesep}"
    )
    return arm_str


def filter_signal(
    data: npt.ArrayLike,
    lowcut: float,
    highcut: float,
    sample_rate: float,
    order: int = 3,
    axis: int = 0,
) -> npt.ArrayLike:
    """Apply band-pass filter to signal using butterworth design and forward-backward cascaded filter

    Args:
        data (npt.ArrayLike): Signal
        lowcut (float): Lower cutoff in Hz
        highcut (float): Upper cutoff in Hz
        sample_rate (float): Sampling rate in Hz
        order (int, optional): Filter order. Defaults to 3.

    Returns:
        npt.ArrayLike: Filtered signal
    """
    sos = get_butter_bp_sos(
        lowcut=lowcut, highcut=highcut, sample_rate=sample_rate, order=order
    )
    return scipy.signal.sosfiltfilt(sos, data, axis=axis)


def resample_signal(
    data: npt.ArrayLike, sample_rate: float, target_rate: float, axis: int = 0
) -> npt.ArrayLike:
    """Resample signal using scipy FFT-based resample routine.

    Args:
        data (npt.ArrayLike): Signal
        sample_rate (float): Signal sampling rate
        target_rate (float): Target sampling rate
        axis (int, optional): Axis to resample along. Defaults to 0.

    Returns:
        npt.ArrayLike: Resampled signal
    """
    desired_length = int(np.round(data.shape[axis] * target_rate / sample_rate))
    return scipy.signal.resample(data, desired_length, axis=axis)


def normalize_signal(
    data: npt.ArrayLike, eps: float = 1e-3, axis: int = 0
) -> npt.ArrayLike:
    """Normalize signal about its mean and std.

    Args:
        data (npt.ArrayLike): Signal
        eps (float, optional): Epsilon added to st. dev. Defaults to 1e-3.
        axis (int, optional): Axis to normalize along. Defaults to 0.

    Returns:
        npt.ArrayLike: Normalized signal
    """
    mu = np.nanmean(data, axis=axis)
    std = np.nanstd(data, axis=axis) + eps
    return (data - mu) / std


def rolling_standardize(x: npt.ArrayLike, win_len: int) -> npt.ArrayLike:
    """Performs rolling standardization

    Args:
        x (npt.ArrayLike): Data
        win_len (int): Window length

    Returns:
        npt.ArrayLike: Standardized data
    """
    x_roll = np.lib.stride_tricks.sliding_window_view(x, win_len)
    x_roll_std = np.std(x_roll, axis=-1)
    x_roll_mu = np.mean(x_roll, axis=-1)
    x_std = np.concatenate(
        (np.repeat(x_roll_std[0], x.shape[0] - x_roll_std.shape[0]), x_roll_std)
    )
    x_mu = np.concatenate(
        (np.repeat(x_roll_mu[0], x.shape[0] - x_roll_mu.shape[0]), x_roll_mu)
    )
    x_norm = (x - x_mu) / x_std
    return x_norm


def running_mean_std(
    iterator, dtype: npt.DTypeLike | None = None
) -> tuple[float, float]:
    """Calculate mean and standard deviation while iterating over the data iterator.
        iterator (Iterable): Data iterator.
        dtype (npt.DTypeLike | None): Type of accumulators.
    Returns:
        tuple[float, float]; mean, Std.
    """
    sum_x = np.zeros((), dtype=dtype)
    sum_x2 = np.zeros((), dtype=dtype)
    n = 0
    for x in iterator:
        sum_x += np.sum(x, dtype=dtype)
        sum_x2 += np.sum(x**2, dtype=dtype)
        n += x.size
    mean = sum_x / n
    std = np.math.sqrt((sum_x2 / n) - (mean**2))
    return mean, std


def pad_signal(
    x: npt.ArrayLike,
    max_len: int | None = None,
    padding: Literal["pre", "post"] = "pre",
) -> npt.ArrayLike:
    """Pads signal shorter than `max_len` and trims those longer than `max_len`.
    Args:
        x (npt.ArrayLike): Array of sequences.
        max_len (int | None, optional): Maximum length of sequence. Defaults to longest.
        padding (Literal["pre", "post"]): Before or after sequence. Defaults to pre.
    Returns:
        npt.ArrayLike Array of padded sequences.
    """
    if max_len is None:
        max_len = max(map(len, x))
    x_shape = x[0].shape
    x_dtype = x[0].dtype
    x_padded = np.zeros((len(x), max_len) + x_shape[1:], dtype=x_dtype)
    for i, x_i in enumerate(x):
        trim_len = min(max_len, len(x_i))
        if padding == "pre":
            x_padded[i, -trim_len:] = x_i[-trim_len:]
        elif padding == "post":
            x_padded[i, :trim_len] = x_i[:trim_len]
        else:
            raise ValueError(f"Unknown padding: {padding}")
    return x_padded
