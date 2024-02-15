import logging
import os
from enum import IntEnum

import numpy as np
import numpy.typing as npt
import tensorflow as tf

from ..defines import HeartRhythm
from .dataset import HKDataset
from .defines import PatientGenerator, SampleGenerator

logger = logging.getLogger(__name__)


class PtbxlRhythm(IntEnum):
    """Icentia rhythm labels"""

    sr = 0
    afib = 1
    stach = 2
    sarrh = 3
    sbrad = 4
    pace = 5
    svarr = 6
    bigu = 7
    aflt = 8
    svtac = 9
    psvt = 10
    trigu = 11


##
# These map PTBXL specific labels to common labels
##
HeartRhythmMap = {
    PtbxlRhythm.sr: HeartRhythm.normal,
    PtbxlRhythm.afib: HeartRhythm.afib,
    PtbxlRhythm.stach: HeartRhythm.normal,
    PtbxlRhythm.sarrh: HeartRhythm.normal,
    PtbxlRhythm.sbrad: HeartRhythm.normal,
    PtbxlRhythm.pace: HeartRhythm.normal,
    PtbxlRhythm.svarr: HeartRhythm.afib,
    PtbxlRhythm.bigu: HeartRhythm.afib,
    PtbxlRhythm.aflt: HeartRhythm.aflut,
    PtbxlRhythm.svtac: HeartRhythm.normal,
    PtbxlRhythm.psvt: HeartRhythm.afib,
    PtbxlRhythm.trigu: HeartRhythm.afib,
}

PtbxlLeadsMap = {
    "i": 0,
    "ii": 1,
    "iii": 2,
    "avr": 3,
    "avl": 4,
    "avf": 5,
    "v1": 6,
    "v2": 7,
    "v3": 8,
    "v4": 9,
    "v5": 10,
    "v6": 11,
}


class PtbxlDataset(HKDataset):
    """PTBXL dataset"""

    def __init__(
        self,
        ds_path: os.PathLike,
        task: str,
        frame_size: int,
        target_rate: int,
        spec: tuple[tf.TensorSpec, tf.TensorSpec],
        class_map: dict[int, int] | None = None,
    ) -> None:
        super().__init__(
            ds_path=ds_path / "ptbxl",
            task=task,
            frame_size=frame_size,
            target_rate=target_rate,
            spec=spec,
            class_map=class_map,
        )

    @property
    def sampling_rate(self) -> int:
        """Sampling rate in Hz"""
        return 500

    @property
    def mean(self) -> float:
        """Dataset mean"""
        return 0

    @property
    def std(self) -> float:
        """Dataset st dev"""
        return 1

    @property
    def patient_ids(self) -> npt.NDArray:
        """Get dataset patient IDs

        Returns:
            npt.NDArray: patient IDs
        """
        return np.arange(1, 21838)

    def get_train_patient_ids(self) -> npt.NDArray:
        """Get dataset training patient IDs

        Returns:
            npt.NDArray: patient IDs
        """
        return self.patient_ids[:18500]

    def get_test_patient_ids(self) -> npt.NDArray:
        """Get dataset patient IDs reserved for testing only

        Returns:
            npt.NDArray: patient IDs
        """
        return self.patient_ids[18500:]

    def task_data_generator(
        self,
        patient_generator: PatientGenerator,
        samples_per_patient: int | list[int] = 1,
    ) -> SampleGenerator:
        """Task-level data generator.

        Args:
            patient_generator (PatientGenerator): Patient data generator
            samples_per_patient (int | list[int], optional): # samples per patient. Defaults to 1.

        Returns:
            SampleGenerator: Sample data generator
        """
        raise NotImplementedError()
