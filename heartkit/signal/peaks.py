import numpy as np
import numpy.typing as npt
import scipy.ndimage as spn


def find_peaks(
    data: npt.NDArray,
    sample_rate: float = 1000,
    qrs_window: float = 0.1,
    avg_window: float = 1.0,
    qrs_prom_weight: float = 1.5,
    qrs_min_len_weight: float = 0.4,
    qrs_min_delay: float = 0.3,
):
    """Find R peaks in ECG signal using QRS gradient method.
    Args:
        data (array): ECG signal.
        sample_rate (float, optional): Sampling rate in Hz. Defaults to 1000 Hz.
        qrs_window (float, optional): Window size in seconds to compute QRS gradient. Defaults to 0.1 s.
        avg_window (float, optional): Window size in seconds to compute average gradient. Defaults to 1.0 s.
        qrs_prom_weight (float, optional): Weight to compute minimum QRS height. Defaults to 1.5.
        qrs_min_len_weight (float, optional): Weight to compute minimum QRS length. Defaults to 0.4.
        qrs_min_delay (float, optional): Minimum delay between QRS complexes. Defaults to 0.3 s.
    Returns:
        npt.NDArray: R peaks.
    """

    # Compute gradient of signal for both QRS and average.
    abs_grad = np.abs(np.gradient(data))
    qrs_kernel = int(np.rint(qrs_window * sample_rate))
    avg_kernel = int(np.rint(avg_window * sample_rate))

    # Smooth gradients
    qrs_grad = spn.uniform_filter1d(abs_grad, qrs_kernel, mode="nearest")
    avg_grad = spn.uniform_filter1d(qrs_grad, avg_kernel, mode="nearest")

    min_qrs_height = qrs_prom_weight * avg_grad

    # Identify start and end of QRS complexes.
    qrs = qrs_grad > min_qrs_height
    beg_qrs = np.where(np.logical_and(np.logical_not(qrs[0:-1]), qrs[1:]))[0]
    end_qrs = np.where(np.logical_and(qrs[0:-1], np.logical_not(qrs[1:])))[0]
    end_qrs = end_qrs[end_qrs > beg_qrs[0]]

    num_qrs = min(beg_qrs.size, end_qrs.size)
    min_qrs_len = np.mean(end_qrs[:num_qrs] - beg_qrs[:num_qrs]) * qrs_min_len_weight
    min_qrs_delay = int(np.rint(qrs_min_delay * sample_rate))

    peaks = []
    for i in range(num_qrs):
        beg, end = beg_qrs[i], end_qrs[i]
        peak = beg + np.argmax(data[beg:end])
        qrs_len = end - beg
        qrs_delay = peak - peaks[-1] if peaks else min_qrs_delay

        # Enforce minimum delay between peaks
        if qrs_delay < min_qrs_delay or qrs_len < min_qrs_len:
            continue
        peaks.append(peak)
    # END FOR

    return np.array(peaks, dtype=int)


def filter_peaks(
    peaks: npt.NDArray,
    sample_rate: float = 1000,
) -> npt.NDArray:
    """Filter out peaks with RR intervals outside of normal range.
    Args:
        peaks (array): R peaks.
        sample_rate (float, optional): Sampling rate in Hz. Defaults to 1000 Hz.
    Returns:
        npt.NDArray: Filtered peaks.
    """
    lowcut = 0.3 * sample_rate
    highcut = 2 * sample_rate

    # Capture RR intervals
    rr_ints = np.diff(peaks)
    rr_ints = np.hstack((rr_ints[0], rr_ints))

    # Filter out peaks with RR intervals outside of normal range
    rr_mask = np.where((rr_ints < lowcut) | (rr_ints > highcut), 1, 0)

    # Filter out peaks that deviate more than 30%
    rr_mask = quotient_filter_mask(rr_ints, mask=rr_mask, lowcut=0.7, highcut=1.3)
    filt_peaks = peaks[np.where(rr_mask == 0)[0]]
    return filt_peaks


def compute_rr_intervals(
    peaks: npt.NDArray,
    sample_rate: float = 1000,
) -> npt.NDArray:
    """Compute RR intervals from R peaks.
    Args:
        peaks (array): R peaks.
        sample_rate (float, optional): Sampling rate in Hz. Defaults to 1000 Hz.
    Returns:
        npt.NDArray: RR intervals.
    """

    rr_ints = np.diff(peaks)
    if rr_ints.size == 0:
        return rr_ints
    rr_ints = np.hstack((rr_ints[0], rr_ints))
    return rr_ints


def filter_rr_intervals(
    rr_ints: npt.NDArray, sample_rate: float = 1000, min_rr: float = 0.3, max_rr: float = 2.0, min_delta: float = 0.3
) -> npt.NDArray:
    """Filter out peaks with RR intervals outside of normal range.
    Args:
        rr_ints (array): RR intervals.
        sample_rate (float, optional): Sampling rate in Hz. Defaults to 1000 Hz.
        min_rr (float, optional): Minimum RR interval in seconds. Defaults to 0.3 s.
        max_rr (float, optional): Maximum RR interval in seconds. Defaults to 2.0 s.
        min_delta (float, optional): Minimum RR interval delta. Defaults to 0.3.
    Returns:
        npt.NDArray: Filtered RR intervals.
    """
    if rr_ints.size == 0:
        return []

    # Filter out peaks with RR intervals outside of normal range
    rr_mask = np.where((rr_ints < min_rr * sample_rate) | (rr_ints > max_rr * sample_rate), 1, 0)

    # Filter out peaks that deviate more than delta
    rr_mask = quotient_filter_mask(rr_ints, mask=rr_mask, lowcut=1 - min_delta, highcut=1 + min_delta)

    return rr_mask


def quotient_filter_mask(
    data: npt.NDArray, mask: npt.NDArray | None = None, iterations: int = 2, lowcut: float = 0.8, highcut: float = 1.2
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
        if filt_idxs.size <= 1:
            break
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
