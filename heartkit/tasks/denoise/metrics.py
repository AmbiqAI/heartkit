import numpy as np
import numpy.typing as npt


def ssd(y_true: npt.NDArray, y_pred: npt.NDArray, axis: int = 1) -> npt.NDArray:
    """Sum of squared distance

    Args:
        y_true (npt.NDArray): True values
        y_pred (npt.NDArray): Predicted values
        axis (int, optional): Axis to sum. Defaults to 1.

    Returns:
        npt.NDArray: Sum of squared distance
    """
    return np.sum(np.square(y_true - y_pred), axis=axis)


def mad(y_true: npt.NDArray, y_pred: npt.NDArray, axis: int = 1) -> npt.NDArray:
    """Absolute max difference

    Args:
        y_true (npt.NDArray): True values
        y_pred (npt.NDArray): Predicted values
        axis (int, optional): Axis to sum. Defaults to 1.

    Returns:
        npt.NDArray: Absolute max difference
    """
    return np.max(np.abs(y_true - y_pred), axis=axis)


def prd(y_true: npt.NDArray, y_pred: npt.NDArray, axis: int = 1) -> npt.NDArray:
    """Percentage root mean square difference

    Args:
        y_true (npt.NDArray): True values
        y_pred (npt.NDArray): Predicted values
        axis (int, optional): Axis to sum. Defaults to 1.

    Returns:
        npt.NDArray: Percentage root mean square difference
    """
    N = np.sum(np.square(y_pred - y_true), axis=axis)
    D = np.sum(np.square(y_pred - np.mean(y_true)), axis=axis)
    PRD = np.sqrt(N / D) * 100

    return PRD


def snr(y1: npt.NDArray, y2: npt.NDArray) -> npt.NDArray:
    """Compute signal to noise ratio

    Args:
        y1 (npt.NDArray): True values
        y2 (npt.NDArray): Predicted values

    Returns:
        npt.NDArray: Signal to noise ratio
    """
    N = np.sum(np.square(y1), axis=1)
    D = np.sum(np.square(y2 - y1), axis=1)

    SNR = 10 * np.log10(N / D)

    return SNR


def SNR_improvement(y_in: npt.NDArray, y_out: npt.NDArray, y_clean: npt.NDArray) -> npt.NDArray:
    """Compute signal to noise ratio improvement

    Args:
        y_in (npt.NDArray): Input signal
        y_out (npt.NDArray): Output signal
        y_clean (npt.NDArray): Clean signal

    Returns:
        npt.NDArray: Signal to noise ratio improvement
    """
    return snr(y_clean, y_out) - snr(y_clean, y_in)
