import abc
import contextlib
import logging
import os
from pathlib import Path
from typing import Generator

import numpy.typing as npt
import sklearn.model_selection

from .defines import PatientGenerator, PatientData

logger = logging.getLogger(__name__)


class HKDataset(abc.ABC):
    path: Path
    _cacheable: bool
    _cached_data: dict[str, npt.NDArray]

    def __init__(self, path: os.PathLike, cacheable: bool = True) -> None:
        """HKDataset serves as a base class to download and provide unified access to datasets.

        Args:
            path (os.PathLike): Path to dataset
            cacheable (bool, optional): If dataset supports file caching. Defaults

        Example:

        ```python
        import numpy as np
        import heartkit as hk

        class MyDataset(hk.HKDataset):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)

            @property
            def name(self) -> str:
                return 'my-dataset'

            @property
            def sampling_rate(self) -> int:
                return 100

            def get_train_patient_ids(self) -> npt.NDArray:
                return np.arange(80)

            def get_test_patient_ids(self) -> npt.NDArray:
                return np.arange(80, 100)

            @contextlib.contextmanager
            def patient_data(self, patient_id: int) -> Generator[PatientData, None, None]:
                data = np.random.randn(1000)
                segs = np.random.randint(0, 1000, (10, 2))
                yield {"data": data, "segmentations": segs}

            def signal_generator(
                self,
                patient_generator: PatientGenerator,
                frame_size: int,
                samples_per_patient: int = 1,
                target_rate: int | None = None,
            ) -> Generator[npt.NDArray, None, None]:
                for patient in patient_generator:
                    for _ in range(samples_per_patient):
                        with self.patient_data(patient) as pt:
                            yield pt["data"]

            def download(self, num_workers: int | None = None, force: bool = False):
                pass

        # Register dataset
        hk.DatasetFactory.register("my-dataset", MyDataset)
        ```
        """
        self.path = Path(path)
        self._cacheable = cacheable
        self._cached_data = {}

    @property
    def name(self) -> str:
        """Dataset name"""
        return self.path.stem

    @property
    def cacheable(self) -> bool:
        """If dataset supports in-memory caching.

        On smaller datasets, it is recommended to cache the entire dataset in memory.
        """
        return self._cacheable

    @cacheable.setter
    def cacheable(self, value: bool):
        """Set if in-memory caching is enabled"""
        self._cacheable = value

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
        """Get dataset's defined training patient IDs

        Returns:
            npt.NDArray: patient IDs
        """
        raise NotImplementedError()

    def get_test_patient_ids(self) -> npt.NDArray:
        """Get dataset's patient IDs reserved for testing only

        Returns:
            npt.NDArray: patient IDs
        """
        raise NotImplementedError()

    @contextlib.contextmanager
    def patient_data(self, patient_id: int) -> Generator[PatientData, None, None]:
        """Get patient data

        Args:
            patient_id (int): Patient ID

        Returns:
            Generator[PatientData, None, None]: Patient data
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
            patient_generator (PatientGenerator): Generator that yields patient data.
            frame_size (int): Frame size
            samples_per_patient (int, optional): Samples per patient. Defaults to 1.
            target_rate (int | None, optional): Target rate. Defaults to None.

        Returns:
            Generator[npt.NDArray, None, None]: Generator sample of data
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
