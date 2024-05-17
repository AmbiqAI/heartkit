import abc
import contextlib
import logging
import os
from pathlib import Path
from typing import Generator

import h5py
import numpy.typing as npt
import sklearn

from .defines import PatientGenerator

logger = logging.getLogger(__name__)


class HKDataset(abc.ABC):
    """HeartKit dataset base class"""

    ds_path: Path

    def __init__(self, ds_path: os.PathLike) -> None:
        """HeartKit dataset base class"""
        self.ds_path = Path(ds_path)

    @property
    def name(self) -> str:
        """Dataset name"""
        return self.ds_path.stem

    @property
    def cachable(self) -> bool:
        """If dataset supports file caching."""
        return True

    @property
    def sampling_rate(self) -> int:
        """Sampling rate in Hz"""
        return 0

    @property
    def mean(self) -> float:
        """Dataset mean"""
        return 0

    @property
    def std(self) -> float:
        """Dataset st dev"""
        return 1

    def get_train_patient_ids(self) -> npt.NDArray:
        """Get training patient IDs

        Returns:
            npt.NDArray: patient IDs
        """
        raise NotImplementedError()

    def get_test_patient_ids(self) -> npt.NDArray:
        """Get patient IDs reserved for testing only

        Returns:
            npt.NDArray: patient IDs
        """
        raise NotImplementedError()

    @contextlib.contextmanager
    def patient_data(self, patient_id: int) -> Generator[h5py.Group, None, None]:
        """Get patient data

        Args:
            patient_id (int): Patient ID

        Returns:
            Generator[h5py.Group, None, None]: Patient data
        """
        raise NotImplementedError()

    def signal_generator(
        self,
        patient_generator: PatientGenerator,
        frame_size: int,
        samples_per_patient: int = 1,
        target_rate: int | None = None,
    ) -> Generator[npt.NDArray, None, None]:
        """Generate random frames.

        Args:
            patient_generator (PatientGenerator): Generator that yields a tuple of patient id and patient data.
                    Patient data may contain only signals, since labels are not used.
            samples_per_patient (int): Samples per patient.

        Returns:
            Generator[npt.NDArray, None, None]: Generator of input data of shape (frame_size, 1)
        """
        raise NotImplementedError()

    def download(self, num_workers: int | None = None, force: bool = False):
        """Download dataset

        Args:
            num_workers (int | None, optional): # parallel workers. Defaults to None.
            force (bool, optional): Force redownload. Defaults to False.
        """
        raise NotImplementedError()

    def split_train_test_patients(
        self,
        patient_ids: npt.NDArray,
        test_size: float,
        label_map: dict[int, int] | None = None,
        label_type: str | None = None,
    ) -> list[list[int]]:
        """Perform train/test split on patients for given task.
        NOTE: We only perform inter-patient splits and not intra-patient.

        Args:
            patient_ids (npt.NDArray): Patient Ids
            test_size (float): Test size
            label_map (dict[int, int], optional): Label map. Defaults to None.
            label_type (str, optional): Label type. Defaults to None.

        Returns:
            list[list[int]]: Train and test sets of patient ids
        """
        return sklearn.model_selection.train_test_split(patient_ids, test_size=test_size)

    def filter_patients_for_labels(
        self, patient_ids: npt.NDArray, label_map: dict[int, int] | None = None, label_type: str | None = None
    ) -> npt.NDArray:
        """Filter patients for given labels.

        Args:
            patient_ids (npt.NDArray): Patient ids
            label_map (dict[int, int], optional): Label map. Defaults to None.
            label_type (str, optional): Label type. Defaults to None.

        Returns:
            npt.NDArray: Filtered patient ids
        """
        return patient_ids

    def close(self):
        """Close dataset"""
