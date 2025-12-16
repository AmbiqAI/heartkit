import os
from pathlib import Path

import h5py
import numpy as np
import numpy.typing as npt
import physiokit as pk
import helia_edge as helia

logger = helia.utils.setup_logger(__name__)


class NstdbNoise:
    def __init__(
        self,
        target_rate: int,
    ):
        """Noise stress test database (NSTDB) noise generator.

        Args:
            target_rate (int): Target rate in Hz
        """

        self.target_rate = target_rate
        self._noises: dict[str, npt.NDArray] | None = None

    @property
    def sampling_rate(self) -> int:
        """Sampling rate in Hz"""
        return 360

    def set_target_rate(self, target_rate: int):
        """Set target rate."""
        if target_rate == self.target_rate:
            return
        self._noises = None
        self.target_rate = target_rate

    def _load_noise_data(self):
        """Load noise data from HDF5 file."""
        logger.debug("Loading noise data from HDF5 file.")
        _file_path = os.path.realpath(__file__)
        noise_path = Path(_file_path).parent.parent / "assets" / "data" / "nstdb.h5"
        with h5py.File(noise_path, "r") as f:
            bw = pk.signal.resample_signal(f["bw"][:], self.sampling_rate, self.target_rate, axis=0)
            ma = pk.signal.resample_signal(f["ma"][:], self.sampling_rate, self.target_rate, axis=0)
            em = pk.signal.resample_signal(f["em"][:], self.sampling_rate, self.target_rate, axis=0)
            self._noises = {"bw": bw, "ma": ma, "em": em}
        # END WITH

    def get_noise(self, noise_type: str) -> npt.NDArray:
        """Get noise data of type.

        Args:
            noise_type (str): Noise type
        """
        if self._noises is None:
            self._load_noise_data()
        # END IF
        return self._noises[noise_type]

    def apply_noise(self, data: npt.NDArray, noise_level: float, axis: int = 0) -> npt.NDArray:
        """Add noise to ECG signal.

        Args:
            data (npt.NDArray): ECG signal
            noise_level (float): Noise level
            axis (int, optional): Axis to apply noise. Defaults to 0.

        Returns:
            npt.NDArray: Noisy ECG signal
        """

        if self._noises is None:
            self._load_noise_data()
        # END IF

        frame_size = data.shape[axis]
        bw = self._noises["bw"]
        ma = self._noises["ma"]
        em = self._noises["em"]

        # NOTE: If signal is too long, apply patches across

        bw_lead = np.random.randint(0, bw.shape[1])
        bw_start = np.random.randint(bw.shape[0] - frame_size)
        bw_end = bw_start + frame_size

        ma_lead = np.random.randint(0, ma.shape[1])
        ma_start = np.random.randint(ma.shape[0] - frame_size)
        ma_end = ma_start + frame_size

        em_lead = np.random.randint(0, em.shape[1])
        em_start = np.random.randint(em.shape[0] - frame_size)
        em_end = em_start + frame_size

        bw_amp = np.abs(np.random.normal(0, noise_level))
        ma_amp = np.abs(np.random.normal(0, noise_level))
        em_amp = np.abs(np.random.normal(0, noise_level))

        data += bw_amp * bw[bw_start:bw_end, bw_lead]
        data += ma_amp * ma[ma_start:ma_end, ma_lead]
        data += em_amp * em[em_start:em_end, em_lead]

        return data

    def close(self):
        """Close noise generator."""
        self._noises = None
