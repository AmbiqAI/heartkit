import functools
from typing import Generator, Literal, Optional, Tuple

import numpy as np
import numpy.typing as npt
import tensorflow as tf
from scipy.signal import butter, sosfiltfilt

# if normalize:
#     data_generator = map(
#         lambda x_y: (normalize(x_y[0]), x_y[1]), data_generator
#     )
# def normalize(
#     self, array: npt.ArrayLike, local: bool = True, filter_enable: bool = False
# ) -> npt.ArrayLike:
#     """Normalize an array using the mean and standard deviation calculated over the entire dataset.

#     Args:
#         array (npt.ArrayLike):  Numpy array to normalize
#         inplace (bool, optional): Whether to perform the normalization steps in-place. Defaults to False.
#         local (bool, optional): Local mean and std or global. Defaults to True.
#         filter_enable (bool, optional): Enable band-pass filter. Defaults to False.

#     Returns:
#         npt.ArrayLike: Normalized array
#     """
#     if filter_enable:
#         filt_array = butter_bp_filter(
#             array, lowcut=0.5, highcut=40, sample_rate=self.sampling_rate, order=2
#         )
#     else:
#         filt_array = np.copy(array)
#     with warnings.catch_warnings():
#         warnings.simplefilter("ignore")
#         filt_array = sklearn.preprocessing.scale(
#             filt_array, with_mean=True, with_std=True, copy=False
#         )
#     return filt_array


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
