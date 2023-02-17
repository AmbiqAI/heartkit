import functools
from typing import Generator, Literal, Optional, Tuple

import numpy as np
import numpy.typing as npt
import scipy.signal
import tensorflow as tf


def preprocess_signal(data: npt.ArrayLike, sample_rate: float, target_rate: float):
    """Pre-process signal

    Args:
        data (npt.ArrayLike): Signal
        sample_rate (float): Sampling rate (Hz)
        target_rate (float): Target sampling rate (Hz)

    Returns:
        _type_: _description_
    """
    axis = 0
    norm_eps = 0.1
    filt_lo = 0.5
    filt_hi = 40

    y = filter_signal(
        data, lowcut=filt_lo, highcut=filt_hi, sample_rate=sample_rate, axis=axis
    )
    if sample_rate != target_rate:
        y = resample_signal(
            y, sample_rate=sample_rate, target_rate=target_rate, axis=axis
        )
    y = normalize_signal(y, eps=norm_eps)
    return y


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
    std = np.nanstd(data, axis=axis)
    if eps != 0:
        std += eps
    y = np.copy(data)
    y -= mu
    y /= std
    return y


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
    iterator, dtype: Optional[npt.DTypeLike] = None
) -> Tuple[float, float]:
    """Calculate mean and standard deviation while iterating over the data iterator.
        iterator (Iterable): Data iterator.
        dtype (Optional[npt.DTypeLike]): Type of accumulators.
    Returns:
        Tuple[float, float]; mean, Std.
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


def numpy_dataset_generator(
    x: npt.ArrayLike, y: npt.ArrayLike
) -> Generator[Tuple[npt.ArrayLike, npt.ArrayLike], None, None]:
    """Create generator from numpy dataset where first axis is samples

    Args:
        x (npt.ArrayLike): X data
        y (npt.ArrayLike): Y data

    Yields:
        Generator[Tuple[npt.ArrayLike, npt.ArrayLike], None, None]: Samples
    """
    for i in range(x.shape[0]):
        yield x[i], y[i]


def create_dataset_from_data(
    x: npt.ArrayLike, y: npt.ArrayLike, spec: Tuple[tf.TensorSpec]
):
    """Helper function to create dataset from static data
    Args:
        x (npt.ArrayLike): Numpy data
        y (npt.ArrayLike): Numpy labels
    """
    gen = functools.partial(numpy_dataset_generator, x=x, y=y)
    dataset = tf.data.Dataset.from_generator(generator=gen, output_signature=spec)
    return dataset


def pad_sequences(
    x: npt.ArrayLike,
    max_len: Optional[int] = None,
    padding: Literal["pre", "post"] = "pre",
) -> npt.ArrayLike:
    """Pads sequences shorter than `max_len` and trims those longer than `max_len`.
    Args:
        x (npt.ArrayLike): Array of sequences.
        max_len (Optional[int], optional): Maximum length of sequence. Defaults to longest.
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
