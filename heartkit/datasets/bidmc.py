import contextlib
import functools
import logging
import random
from typing import Generator

import h5py
import numpy as np
import numpy.typing as npt
import physiokit as pk

from .dataset import HKDataset
from .defines import PatientGenerator

logger = logging.getLogger(__name__)


BidmcLeadsMap = {"ii": 0}


class BidmcDataset(HKDataset):
    """BIDMC dataset"""

    def __init__(
        self,
        leads: list[int] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.leads = leads or list(BidmcLeadsMap.values())

    @property
    def name(self) -> str:
        """Dataset name"""
        return "bidmc"

    @property
    def sampling_rate(self) -> int:
        """Sampling rate in Hz"""
        return 125

    @property
    def mean(self) -> float:
        """Dataset mean"""
        return 0

    @property
    def std(self) -> float:
        """Dataset st dev"""
        return 1

    @functools.cached_property
    def patient_ids(self) -> npt.NDArray:
        """Get dataset patient IDs

        Returns:
            npt.NDArray: patient IDs
        """
        pts = np.arange(1, 54)
        return pts

    def get_train_patient_ids(self) -> npt.NDArray:
        """Get dataset training patient IDs

        Returns:
            npt.NDArray: patient IDs
        """
        idx = int(len(self.patient_ids) * 0.80)
        return self.patient_ids[:idx]

    def get_test_patient_ids(self) -> npt.NDArray:
        """Get dataset patient IDs reserved for testing only

        Returns:
            npt.NDArray: patient IDs
        """
        idx = int(len(self.patient_ids) * 0.80)
        return self.patient_ids[idx:]

    def _pt_key(self, patient_id: int):
        """Get patient key"""
        return f"bidmc{patient_id:02d}"

    @contextlib.contextmanager
    def patient_data(self, patient_id: int) -> Generator[h5py.Group, None, None]:
        """Get patient data

        Args:
            patient_id (int): Patient ID

        Returns:
            Generator[h5py.Group, None, None]: Patient data
        """
        with h5py.File(self.path / f"{self._pt_key(patient_id)}.h5", mode="r") as h5:
            yield h5

    def signal_generator(
        self,
        patient_generator: PatientGenerator,
        frame_size: int,
        samples_per_patient: int = 1,
        target_rate: int | None = None,
    ) -> Generator[npt.NDArray, None, None]:
        """Generate random frames.

        Args:
            patient_generator (PatientGenerator): Patient Generator
            frame_size (int): Frame size
            samples_per_patient (int, optional): Samples per patient. Defaults to 1.
            target_rate (int, optional): Target rate. Defaults to None.

        Returns:
            Generator[npt.NDArray, None, None]: Generator of input data
        """
        if target_rate is None:
            target_rate = self.sampling_rate

        input_size = int(np.ceil((self.sampling_rate / target_rate) * frame_size))
        for pt in patient_generator:
            with self.patient_data(pt) as h5:
                data: h5py.Dataset = h5["data"][:]
            # END WITH
            for _ in range(samples_per_patient):
                lead = random.choice(self.leads)
                start = np.random.randint(0, data.shape[1] - input_size)
                x = data[lead, start : start + input_size].squeeze()
                x = np.nan_to_num(x).astype(np.float32)
                if self.sampling_rate != target_rate:
                    x = pk.signal.resample_signal(x, self.sampling_rate, target_rate, axis=0)
                    x = x[:frame_size]
                # END IF
                yield x
            # END FOR
        # END FOR

    def download(self, num_workers: int | None = None, force: bool = False):
        """Download dataset

        This will download preprocessed HDF5 files from S3.

        Args:
            num_workers (int | None, optional): # parallel workers. Defaults to None.
            force (bool, optional): Force redownload. Defaults to False.
        """
        raise NotImplementedError("Download not implemented for BIDMC dataset")
