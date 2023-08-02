"""Add various noise sources to signal."""
import neurokit2 as nk
import numpy.typing as npt


def add_baseline_wander(
    data: npt.NDArray,
    amplitude: float = 0.1,
    frequency: float = 0.05,
    sample_rate: int = 1000,
) -> npt.NDArray:
    """Add baseline wander to signal.

    Args:
        data (npt.NDArray): Signal
        amplitude (float, optional): Amplitude in st dev. Defaults to 0.1.
        frequency (float, optional): Baseline wander frequency. Defaults to 0.05 Hz.
        sample_rate (int, optional): Sample rate in Hz. Defaults to 1000 Hz.

    Returns:
        npt.NDArray: Signal w/ baseline wander
    """
    return nk.signal_distort(
        data,
        sampling_rate=sample_rate,
        noise_amplitude=amplitude,
        noise_frequency=frequency,
        silent=True,
    )


def add_motion_noise(
    data: npt.NDArray,
    amplitude: float = 0.2,
    frequency: float = 0.5,
    sample_rate: int = 1000,
) -> npt.NDArray:
    """Add motion noise to signal.

    Args:
        data (npt.NDArray): Signal
        amplitude (float, optional): Amplitude in st dev. Defaults to 0.2.
        frequency (float, optional): Motion frequency in Hz. Defaults to 0.5 Hz.
        sample_rate (int, optional): Sample rate in Hz. Defaults to 1000 Hz.

    Returns:
        npt.NDArray: Signal w/ motion noise
    """
    return nk.signal_distort(
        data,
        sampling_rate=sample_rate,
        noise_amplitude=amplitude,
        noise_frequency=frequency,
        silent=True,
    )


def add_burst_noise(
    data: npt.NDArray,
    amplitude: float = 1,
    frequency: float = 100,
    burst_number: int = 1,
    sample_rate: int = 1000,
) -> npt.NDArray:
    """Add high frequency burst noise to signal.

    Args:
        data (npt.NDArray): Signal
        amplitude (float, optional): Amplitude in st dev. Defaults to 1.
        frequency (float, optional): High frequency burst in Hz. Defaults to 100 Hz.
        burst_number (int, optional): # bursts to inject. Defaults to 1.
        sample_rate (int, optional): Sample rate in Hz. Defaults to 1000 Hz.

    Returns:
        npt.NDArray: Signal w/ burst noise
    """
    return nk.signal_distort(
        data,
        sampling_rate=sample_rate,
        artifacts_amplitude=amplitude,
        artifacts_frequency=frequency,
        artifacts_number=burst_number,
        silent=True,
    )


def add_powerline_noise(
    data: npt.NDArray,
    amplitude: float = 0.01,
    frequency: float = 50,
    sample_rate: int = 1000,
) -> npt.NDArray:
    """Add powerline noise to signal.

    Args:
        data (npt.NDArray): Signal
        amplitude (float, optional): Amplitude in st dev. Defaults to 0.01.
        frequency (float, optional): Powerline frequency in Hz. Defaults to 50 Hz.
        sample_rate (int, optional): Sample rate in Hz. Defaults to 1000 Hz.
    Returns:
        npt.NDArray: Signal w/ powerline noise
    """
    return nk.signal_distort(
        data,
        sampling_rate=sample_rate,
        powerline_amplitude=amplitude,
        powerline_frequency=frequency,
        silent=True,
    )


def add_noise_sources(
    data: npt.NDArray,
    amplitudes: list[float],
    frequencies: list[float],
    sample_rate: int = 1000,
) -> npt.NDArray:
    """Add multiple noise sources to signal.

    Args:
        data (npt.NDArray): Signal
        amplitudes (list[float]): Amplitudes in st dev.
        frequencies (list[float]): Frequencies in Hz.
        sample_rate (int, optional): Sample rate in Hz. Defaults to 1000 Hz.
    Returns:
        npt.NDArray: Signal w/ noise
    """
    return nk.signal_distort(
        data,
        sampling_rate=sample_rate,
        noise_amplitude=amplitudes,
        noise_frequency=frequencies,
        silent=True,
    )
