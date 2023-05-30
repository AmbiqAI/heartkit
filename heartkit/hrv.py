import neurokit2 as nk
import numpy as np
import numpy.typing as npt

from .defines import HeartExportParams, HeartTestParams, HeartTrainParams


def find_peaks_from_segments(
    data: npt.NDArray, qrs_mask: npt.NDArray, sampling_rate: float = 250
) -> npt.NDArray:
    """Find R peaks using QRS segment mask.
    Args:
        data (npt.NDArray): 1-ch ECG data
        qrs_mask (npt.NDArray): Segmentation mask
        sampling_rate (float): Sampling rate in Hz
    Returns:
        npt.NDArray: R peak indices
    """
    # Apply moving average and theshold above 66%
    r_segs = np.where(np.convolve(qrs_mask, np.ones(6), "same") > 4, 1, 0)
    # Locate start and end of QRS segments
    qrs_starts = np.where(np.diff(r_segs) == 1)[0]
    qrs_ends = np.where(np.diff(r_segs) == -1)[0]
    # For now use middle of QRS as R peak. Later look for abs peak in data
    peaks = np.vstack((qrs_starts, qrs_ends)).mean(axis=0).astype(int)
    return peaks


def ecg_rate(
    peaks: npt.NDArray, sampling_rate: float = 1000, desired_length: int | None = None
) -> npt.NDArray:
    """Compute ECG rate from R peak indices

    Args:
        peaks (npt.NDArray): R peak indices
        sampling_rate (float, optional): Sampling rate in Hz. Defaults to 1000.
        desired_length (int | None, optional): Perform extrapolation to given length. Defaults to None.

    Returns:
        npt.NDArray: RR rates
    """
    if np.size(peaks) <= 3:
        return np.zeros_like(peaks)
    rr_intervals = nk.signal_period(
        peaks=peaks, sampling_rate=sampling_rate, desired_length=desired_length
    )
    # NK used global average for first peak- Instead lets take neighboring peak
    rr_intervals[0] = rr_intervals[1]
    return rr_intervals


def ecg_bpm(
    peaks: npt.NDArray,
    sampling_rate: float = 1000,
    min_rate: float | None = None,
    max_rate: float | None = None,
) -> float:
    """Compute average hearte rate (BPM) from R peaks

    Args:
        peaks (npt.NDArray): R peaks
        sampling_rate (float, optional): Sampling rate in Hz. Defaults to 1000.
        min_rate (float | None, optional): Min rate in sec. Defaults to None.
        max_rate (float | None, optional): Max rate in sec. Defaults to None.

    Returns:
        float: Heart rate in BPM
    """
    if np.size(peaks) <= 1:
        return -1
    rr_intervals = ecg_rate(
        peaks=peaks, sampling_rate=sampling_rate, desired_length=None
    )
    if min_rate is not None:
        rr_intervals = np.where(rr_intervals < min_rate, min_rate, rr_intervals)
    if max_rate is not None:
        rr_intervals = np.where(rr_intervals > max_rate, max_rate, rr_intervals)
    if np.size(peaks) <= 1:
        return -1
    return 60 / np.mean(rr_intervals)


def compute_hrv(
    data: npt.NDArray, qrs_mask: npt.NDArray, sampling_rate: int = 1000
) -> tuple[float, npt.NDArray, npt.NDArray]:
    """Compute HRV metrics

    Args:
        data (npt.NDArray): ECG Data
        qrs_mask (npt.NDArray): QRS binary mask
        sampling_rate (int, optional): Sampling rate in Hz. Defaults to 1000.

    Returns:
        tuple[float, npt.NDArray, npt.NDArray]: HR in bpm, RR lengths, R peak indices
    """
    rpeaks = find_peaks_from_segments(
        data=data, qrs_mask=qrs_mask, sampling_rate=sampling_rate
    )
    rr_lens = ecg_rate(peaks=rpeaks, sampling_rate=sampling_rate)
    hr_bpm = ecg_bpm(peaks=rpeaks, sampling_rate=sampling_rate)
    return hr_bpm, rr_lens, rpeaks


def train_model(params: HeartTrainParams):
    """Train HRV model.

    Args:
        params (HeartTrainParams): Training parameters
    """
    # Load segmentation datasets
    # Load segmentation model
    # Compute HRV metrics across true and predicted segmentation masks
    # Routine
    #   Identify R peak from QRS segments and compute mean and std
    #   Remove outlier R peak regions based on confidence and std
    #   Compute heart rate, rhythm label, RR interval, RR variation


def evaluate_model(params: HeartTestParams):
    """Test HRV model.

    Args:
        params (HeartTestParams): Testing/evaluation parameters
    """


def export_model(params: HeartExportParams):
    """Export segmentation model.

    Args:
        params (HeartExportParams): Deployment parameters
    """
