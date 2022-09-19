
import numpy.typing as npt
from scipy.signal import butter, sosfiltfilt

def filter_ecg_signal(data: npt.NDArray, lowcut: float, highcut: float, sample_rate: float, order: int = 2):
    nyq = 0.5 * sample_rate
    low = lowcut / nyq
    high = highcut / nyq
    sos = butter(order, [low, high], btype='band', output='sos')
    f_data = sosfiltfilt(sos, data, axis=0)
    return f_data
