import contextlib
import functools
import os
import random
import zipfile
from enum import IntEnum
from typing import Generator

import h5py
import numpy as np
import numpy.typing as npt
import physiokit as pk
import sklearn.model_selection
import sklearn.preprocessing
from tqdm.contrib.concurrent import process_map

import helia_edge as helia

from .dataset import HKDataset
from .defines import PatientGenerator, PatientData

logger = helia.utils.setup_logger(__name__)


class IcentiaMiniRhythm(IntEnum):
    """Icentia rhythm labels"""

    normal = 1
    afib = 2
    aflut = 3
    end = 4


class IcentiaMiniBeat(IntEnum):
    """Incentia mini beat labels"""

    normal = 1
    pac = 2
    aberrated = 3
    pvc = 4


IcentiaMiniLeadsMap = {
    "i": 0,  # Modified lead I
}


class IcentiaMiniDataset(HKDataset):
    """Icentia-mini dataset"""

    def __init__(
        self,
        leads: list[int] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.leads = leads or list(IcentiaMiniLeadsMap.values())
        self._cached_data = {}

    @property
    def name(self) -> str:
        """Dataset name"""
        return "icentia_mini"

    @property
    def sampling_rate(self) -> int:
        """Sampling rate in Hz"""
        return 250

    @property
    def mean(self) -> float:
        """Dataset mean"""
        return 0.0018

    @property
    def std(self) -> float:
        """Dataset st dev"""
        return 1.3711

    @property
    def patient_ids(self) -> npt.NDArray:
        """Get dataset patient IDs

        Returns:
            npt.NDArray: patient IDs
        """
        return np.arange(11_000)

    def get_train_patient_ids(self) -> npt.NDArray:
        """Get training patient IDs

        Returns:
            npt.NDArray: patient IDs
        """
        return self.patient_ids[:10_000]

    def get_test_patient_ids(self) -> npt.NDArray:
        """Get patient IDs reserved for testing only

        Returns:
            npt.NDArray: patient IDs
        """
        return self.patient_ids[10_000:]

    def _pt_key(self, patient_id: int):
        return f"p{patient_id:05d}"

    def label_key(self, label_type: str = "rhythm") -> str:
        """Get label key

        Args:
            label_type (str, optional): Label type. Defaults to "rhythm".

        Returns:
            str: Label key
        """
        if label_type == "rhythm":
            return "rlabels"
        if label_type == "beat":
            return "blabels"
        raise ValueError(f"Invalid label type: {label_type}")

    @contextlib.contextmanager
    def patient_data(self, patient_id: int) -> Generator[PatientData, None, None]:
        """Get patient data

        Args:
            patient_id (int): Patient ID

        Returns:
            Generator[h5py.Group, None, None]: Patient data
        """
        h5_path = self.path / "icentia_mini.h5"
        pt_key = self._pt_key(patient_id)
        if self.cacheable:
            if patient_id not in self._cached_data:
                pt_data = {}
                with h5py.File(h5_path, mode="r") as h5:
                    pt = h5[pt_key]
                    pt_data["data"] = pt["data"][:]
                    pt_data["rlabels"] = pt["rlabels"][:]
                    pt_data["blabels"] = pt["blabels"][:]
                self._cached_data[patient_id] = pt_data
            # END IF
            yield self._cached_data[patient_id]
        else:
            with h5py.File(h5_path, mode="r") as h5:
                pt = h5[pt_key]
                yield h5
            # END WITH
        # END IF

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
            SampleGenerator: Generator of input data of shape (frame_size, 1)
        """
        if target_rate is None:
            target_rate = self.sampling_rate

        input_size = int(np.ceil((self.sampling_rate / target_rate) * frame_size))
        for pt in patient_generator:
            with self.patient_data(pt) as segments:
                for _ in range(samples_per_patient):
                    segment = segments[np.random.choice(list(segments.keys()))]
                    segment_size = segment["data"].shape[0]
                    frame_start = np.random.randint(segment_size - input_size)
                    frame_end = frame_start + input_size
                    x = segment["data"][frame_start:frame_end].squeeze()
                    x = np.nan_to_num(x).astype(np.float32)
                    if self.sampling_rate != target_rate:
                        x = pk.signal.resample_signal(x, self.sampling_rate, target_rate, axis=0)
                        x = x[:frame_size]
                    # END IF
                    yield x
                # END FOR
            # END WITH
        # END FOR

    def download(self, num_workers: int | None = None, force: bool = False):
        """Download dataset

        This will download preprocessed HDF5 files from S3.

        Args:
            num_workers (int | None, optional): # parallel workers. Defaults to None.
            force (bool, optional): Force redownload. Defaults to False.
        """
        os.makedirs(self.path, exist_ok=True)
        zip_path = self.path / f"{self.name}.zip"

        did_download = helia.utils.download_s3_file(
            key=f"{self.name}/{self.name}.zip",
            dst=zip_path,
            bucket="ambiq-ai-datasets",
            checksum="size",
        )
        if did_download:
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(self.path)

    def split_train_test_patients(
        self,
        patient_ids: npt.NDArray,
        test_size: float,
        label_map: dict[int, int] | None = None,
        label_type: str | None = None,
    ) -> list[list[int]]:
        """Perform train/test split on patients for given task.

        Args:
            patient_ids (npt.NDArray): Patient Ids
            test_size (float): Test size
            label_map (dict[int, int], optional): Label map. Defaults to None.
            label_type (str, optional): Label type. Defaults to None.

        Returns:
            list[list[int]]: Train and test sets of patient ids
        """
        stratify = None

        if label_map is not None and label_type is not None:
            # Use stratified split for rhythm task
            patients_labels = self.get_patients_labels(patient_ids, label_map=label_map, label_type=label_type)
            # Select random label for stratification or -1 if no labels
            stratify = np.array([random.choice(x) if len(x) > 0 else -1 for x in patients_labels])
            # Remove patients w/o labels
            neg_mask = stratify == -1
            stratify = stratify[~neg_mask]
            patient_ids = patient_ids[~neg_mask]
            num_neg = neg_mask.sum()
            if num_neg > 0:
                logger.debug(f"Removed {num_neg} patients w/ no target class")
            # END IF
        # END IF

        return sklearn.model_selection.train_test_split(
            patient_ids,
            test_size=test_size,
            shuffle=True,
            stratify=stratify,
        )

    def filter_patients_for_labels(
        self,
        patient_ids: npt.NDArray,
        label_map: dict[int, int] | None = None,
        label_type: str | None = None,
    ) -> npt.NDArray:
        """Filter patients based on labels.
        Useful to remove patients w/o labels for task to speed up data loading.

        Args:
            patient_ids (npt.NDArray): Patient ids
            label_map (dict[int, int], optional): Label map. Defaults to None.
            label_type (str, optional): Label type. Defaults to None.

        Returns:
            npt.NDArray: Filtered patient ids
        """
        if label_map is None or label_type is None:
            return patient_ids

        patients_labels = self.get_patients_labels(patient_ids, label_map, label_type)
        # Find any patient with empty list
        label_mask = np.array([len(x) > 0 for x in patients_labels])
        neg_mask = label_mask == -1
        num_neg = neg_mask.sum()
        if num_neg > 0:
            logger.debug(f"Removed {num_neg} of {patient_ids.size} patients w/ no target class")
        return patient_ids[~neg_mask]

    def get_patients_labels(
        self,
        patient_ids: npt.NDArray,
        label_map: dict[int, int],
        label_type: str = "rhythm",
    ) -> list[list[int]]:
        """Get class labels for each patient

        Args:
            patient_ids (npt.NDArray): Patient ids
            label_map (dict[int, int]): Label map
            label_type (str, optional): Label type. Defaults to "rhythm".

        Returns:
            list[list[int]]: List of class labels per patient

        """
        ids = patient_ids.tolist()
        func = functools.partial(self.get_patient_labels, label_map=label_map, label_type=label_type)
        pts_labels = process_map(func, ids)
        return pts_labels

    def get_patient_labels(self, patient_id: int, label_map: dict[int, int], label_type: str = "rhythm") -> list[int]:
        """Get class labels for patient

        Args:
            patient_id (int): Patient id
            label_map (dict[int, int]): Label map
            label_type (str, optional): Label type. Defaults to "rhythm".

        Returns:
            list[int]: List of class labels

        """
        label_key = self.label_key(label_type)
        with self.patient_data(patient_id) as pt:
            mask = pt[label_key][:]
            labels = np.unique(mask)
            labels: list[int] = [label_map[lbl] for lbl in labels if label_map.get(lbl, -1) != -1]
        # END WITH
        return list(labels)

    def close(self):
        """Close dataset"""
        self._cached_data.clear()
