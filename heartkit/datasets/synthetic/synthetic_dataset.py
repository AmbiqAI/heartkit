import logging
import os
import random

import numpy as np
import numpy.typing as npt

from ...defines import HeartSegment, HeartTask
from ..dataset import EcgDataset
from ..defines import PatientGenerator, SampleGenerator
from .defines import EcgPresets, SyntheticSegments
from .rhythm_generator import generate_nsr

logger = logging.getLogger(__name__)


class SyntheticDataset(EcgDataset):
    """Synthetic dataset"""

    def __init__(
        self,
        ds_path: str,
        task: HeartTask = HeartTask.rhythm,
        frame_size: int = 1250,
        target_rate: int = 250,
        num_pts: int = 250,
    ) -> None:
        super().__init__(os.path.join(ds_path, "synthetic"), task, frame_size, target_rate)
        self._num_pts = num_pts

    @property
    def sampling_rate(self) -> int:
        """Sampling rate in Hz"""
        return self.target_rate

    @property
    def mean(self) -> float:
        """Dataset mean"""
        return 0

    @property
    def std(self) -> float:
        """Dataset st dev"""
        return 1

    @property
    def patient_ids(self) -> npt.ArrayLike:
        """Get dataset patient IDs

        Returns:
            npt.ArrayLike: patient IDs
        """
        return np.arange(0, self._num_pts)

    def get_train_patient_ids(self) -> npt.ArrayLike:
        """Get dataset training patient IDs

        Returns:
            npt.ArrayLike: patient IDs
        """
        numel = int(0.80 * self._num_pts)
        return self.patient_ids[:numel]

    def get_test_patient_ids(self) -> npt.ArrayLike:
        """Get dataset patient IDs reserved for testing only

        Returns:
            npt.ArrayLike: patient IDs
        """
        numel = int(0.80 * self._num_pts)
        return self.patient_ids[numel:]

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
        if self.task == HeartTask.segmentation:
            return self.segmentation_generator(
                patient_generator=patient_generator,
                samples_per_patient=samples_per_patient,
            )
        raise NotImplementedError()

    def segmentation_generator(
        self,
        patient_generator: PatientGenerator,
        samples_per_patient: int | list[int] = 1,
    ) -> SampleGenerator:
        """Generate frames and segment labels.

        Args:
            patient_generator (PatientGenerator): Patient Generator
            samples_per_patient (int | list[int], optional): # samples per patient. Defaults to 1.
        Returns:
            SampleGenerator: Sample generator
        Yields:
            Iterator[SampleGenerator]
        """
        start_offset = self.sampling_rate
        num_leads = 12  # Use all 12 leads
        presets = (
            EcgPresets.SR,
            EcgPresets.LAHB,
            EcgPresets.LPHB,
            EcgPresets.LBBB,
            EcgPresets.ant_STEMI,
            EcgPresets.random_morphology,
            EcgPresets.high_take_off,
        )
        preset_weights = (20, 1, 1, 1, 1, 1, 1)

        for _ in patient_generator:
            _, syn_ecg, syn_segs_t, _, _ = generate_nsr(
                leads=num_leads,
                signal_frequency=self.sampling_rate,
                rate=np.random.uniform(40, 90),
                preset=random.choices(presets, preset_weights, k=1)[0].value,
                noise_multiplier=np.random.uniform(0.5, 0.9),
                impedance=np.random.uniform(0.75, 1.1),
                p_multiplier=np.random.uniform(0.75, 1.1),
                t_multiplier=np.random.uniform(0.75, 1.1),
                duration=max(5, (self.frame_size / self.sampling_rate) * (samples_per_patient / num_leads / 10)),
                voltage_factor=np.random.uniform(275, 325),
            )
            syn_segs = np.zeros_like(syn_segs_t)
            for i in range(syn_segs_t.shape[0]):
                syn_segs[i, np.where((syn_segs_t[i] == SyntheticSegments.tp_overlap))[0]] = HeartSegment.pwave
                syn_segs[i, np.where((syn_segs_t[i] == SyntheticSegments.p_wave))[0]] = HeartSegment.pwave
                syn_segs[i, np.where((syn_segs_t[i] == SyntheticSegments.qrs_complex))[0]] = HeartSegment.qrs
                syn_segs[i, np.where((syn_segs_t[i] == SyntheticSegments.t_wave))[0]] = HeartSegment.twave

            for _ in range(samples_per_patient):
                # Randomly pick an ECG lead and frame
                lead_idx = np.random.randint(syn_ecg.shape[0])
                frame_start = np.random.randint(start_offset, syn_ecg.shape[1] - self.frame_size)
                frame_end = frame_start + self.frame_size
                x = syn_ecg[lead_idx, frame_start:frame_end].astype(np.float32).reshape((self.frame_size, 1))
                y = syn_segs[lead_idx, frame_start:frame_end].astype(np.int32)
                yield x, y
            # END FOR
        # END FOR

    def uniform_patient_generator(
        self,
        patient_ids: npt.ArrayLike,
        repeat: bool = True,
        shuffle: bool = True,
    ) -> PatientGenerator:
        """Yield data for each patient in the array.

        Args:
            patient_ids (pt.ArrayLike): Array of patient ids
            repeat (bool, optional): Whether to repeat generator. Defaults to True.
            shuffle (bool, optional): Whether to shuffle patient ids.. Defaults to True.

        Returns:
            PatientGenerator: Patient generator

        Yields:
            Iterator[PatientGenerator]
        """
        patient_ids = np.copy(patient_ids)
        while True:
            if shuffle:
                np.random.shuffle(patient_ids)
            for patient_id in patient_ids:
                yield patient_id, None
            # END FOR
            if not repeat:
                break
        # END WHILE

    def download(self, num_workers: int | None = None, force: bool = False):
        """Download dataset

        Args:
            ds_path (str): Path to store dataset
            num_workers (int | None, optional): # parallel workers. Defaults to None.
            force (bool, optional): Force redownload. Defaults to False.
        """
        # Nothing to do

    def load_train_datasets(
        self,
        train_patients: float | None = None,
        val_patients: float | None = None,
        train_pt_samples: int | list[int] | None = None,
        val_pt_samples: int | list[int] | None = None,
        val_size: int | None = None,
        val_file: str | None = None,
        num_workers: int = 1,
    ):
        return super().load_train_datasets(
            train_patients, val_patients, train_pt_samples, val_pt_samples, val_size, None, num_workers  # Dont cache
        )
