import logging
import os
import random

import numpy as np
import numpy.typing as npt
import physiokit as pk
import tensorflow as tf

from ..defines import HeartSegment
from .dataset import HKDataset
from .defines import PatientGenerator, SampleGenerator

logger = logging.getLogger(__name__)


class SyntheticDataset(HKDataset):
    """Synthetic dataset"""

    def __init__(
        self,
        ds_path: os.PathLike,
        task: str,
        frame_size: int,
        target_rate: int,
        spec: tuple[tf.TensorSpec, tf.TensorSpec],
        class_map: dict[int, int] | None = None,
        num_pts: int = 250,
    ) -> None:
        super().__init__(
            ds_path=ds_path / "synthetic",
            task=task,
            frame_size=frame_size,
            target_rate=target_rate,
            spec=spec,
            class_map=class_map,
        )
        self._num_pts = num_pts

    @property
    def cachable(self) -> bool:
        """If dataset supports file caching."""
        return False

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
    def patient_ids(self) -> npt.NDArray:
        """Get dataset patient IDs

        Returns:
            npt.NDArray: patient IDs
        """
        return np.arange(0, self._num_pts)

    def get_train_patient_ids(self) -> npt.NDArray:
        """Get dataset training patient IDs

        Returns:
            npt.NDArray: patient IDs
        """
        numel = int(0.80 * self._num_pts)
        return self.patient_ids[:numel]

    def get_test_patient_ids(self) -> npt.NDArray:
        """Get dataset patient IDs reserved for testing only

        Returns:
            npt.NDArray: patient IDs
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
        if self.task == "segmentation":
            return self.segmentation_generator(
                patient_generator=patient_generator,
                samples_per_patient=samples_per_patient,
            )
        if self.task == "denoise":
            return self.denoising_generator(
                patient_generator=patient_generator,
                samples_per_patient=samples_per_patient,
            )
        raise NotImplementedError()

    def signal_generator(self, patient_generator: PatientGenerator, samples_per_patient: int = 1) -> SampleGenerator:
        """Generate frames using patient generator.

        Args:
            patient_generator (PatientGenerator): Generator that yields a tuple of patient id and patient data.
            samples_per_patient (int): Samples per patient.

        Returns:
            SampleGenerator: Generator of input data of shape (frame_size, 1)
        """

        start_offset = 0
        num_leads = 12  # Use all 12 leads
        presets = (
            pk.ecg.EcgPreset.SR,
            pk.ecg.EcgPreset.AFIB,
            pk.ecg.EcgPreset.LAHB,
            pk.ecg.EcgPreset.LPHB,
            pk.ecg.EcgPreset.LBBB,
            pk.ecg.EcgPreset.ant_STEMI,
            pk.ecg.EcgPreset.random_morphology,
            pk.ecg.EcgPreset.high_take_off,
        )
        preset_weights = (14, 1, 1, 1, 1, 1, 1, 1)
        signal_length = max(2 * self.frame_size, int(self.frame_size * samples_per_patient / num_leads))

        for _ in patient_generator:
            syn_ecg, _, _ = pk.ecg.synthesize(
                signal_length=signal_length,
                sample_rate=self.sampling_rate,
                leads=num_leads,
                heart_rate=np.random.uniform(40, 120),
                preset=random.choices(presets, preset_weights, k=1)[0].value,
                impedance=np.random.uniform(1.0, 2.0),
                p_multiplier=np.random.uniform(0.80, 1.2),
                t_multiplier=np.random.uniform(0.80, 1.2),
                noise_multiplier=0,
                # noise_multiplier=np.random.uniform(0.25, 1.0),
                voltage_factor=np.random.uniform(800, 1000),
            )
            for _ in range(samples_per_patient):
                # Randomly pick an ECG lead and frame
                lead_idx = np.random.randint(syn_ecg.shape[0])
                frame_start = np.random.randint(start_offset, syn_ecg.shape[1] - self.frame_size)
                frame_end = frame_start + self.frame_size
                x = syn_ecg[lead_idx, frame_start:frame_end].astype(np.float32).reshape((self.frame_size,))
                yield x
            # END FOR
        # END FOR

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
        start_offset = 0
        num_leads = 12  # Use all 12 leads
        presets = (
            pk.ecg.EcgPreset.SR,
            pk.ecg.EcgPreset.AFIB,
            pk.ecg.EcgPreset.LAHB,
            pk.ecg.EcgPreset.LPHB,
            pk.ecg.EcgPreset.LBBB,
            pk.ecg.EcgPreset.ant_STEMI,
            pk.ecg.EcgPreset.random_morphology,
            pk.ecg.EcgPreset.high_take_off,
        )
        preset_weights = (14, 1, 1, 1, 1, 1, 1, 1)
        signal_length = max(2 * self.frame_size, int(self.frame_size * samples_per_patient / num_leads))

        for _ in patient_generator:
            syn_ecg, syn_segs_t, _ = pk.ecg.synthesize(
                signal_length=signal_length,
                sample_rate=self.sampling_rate,
                leads=num_leads,
                heart_rate=np.random.uniform(40, 120),
                preset=random.choices(presets, preset_weights, k=1)[0].value,
                impedance=np.random.uniform(1.0, 2.0),
                p_multiplier=np.random.uniform(0.80, 1.2),
                t_multiplier=np.random.uniform(0.80, 1.2),
                noise_multiplier=0,
                # noise_multiplier=np.random.uniform(0.25, 1.0),
                voltage_factor=np.random.uniform(800, 1000),
            )
            syn_segs = np.zeros_like(syn_segs_t)
            for i in range(syn_segs_t.shape[0]):
                syn_segs[i, np.where((syn_segs_t[i] == pk.ecg.EcgSegment.tp_overlap))[0]] = HeartSegment.pwave
                syn_segs[i, np.where((syn_segs_t[i] == pk.ecg.EcgSegment.p_wave))[0]] = HeartSegment.pwave
                syn_segs[i, np.where((syn_segs_t[i] == pk.ecg.EcgSegment.qrs_complex))[0]] = HeartSegment.qrs
                syn_segs[i, np.where((syn_segs_t[i] == pk.ecg.EcgSegment.t_wave))[0]] = HeartSegment.twave
            # END FOR

            for i in range(samples_per_patient):
                # Randomly pick an ECG lead and frame
                lead_idx = np.random.randint(syn_ecg.shape[0])
                frame_start = np.random.randint(start_offset, syn_ecg.shape[1] - self.frame_size)
                frame_end = frame_start + self.frame_size
                x = syn_ecg[lead_idx, frame_start:frame_end].astype(np.float32)
                y = syn_segs[lead_idx, frame_start:frame_end].astype(np.int32)
                y = np.vectorize(self.class_map.get, otypes=[int])(y)
                yield x, y
            # END FOR
        # END FOR

    def denoising_generator(
        self,
        patient_generator: PatientGenerator,
        samples_per_patient: int | list[int] = 1,
    ) -> SampleGenerator:
        """Generate frames and denoised frames."""
        gen = self.signal_generator(patient_generator, samples_per_patient)
        for x in gen:
            x = x.reshape((self.frame_size, 1))
            y = x.copy()
            y = pk.signal.filter_signal(
                y, sample_rate=self.sampling_rate, lowcut=1.0, highcut=30, order=3, forward_backward=True, axis=0
            )
            y = pk.signal.normalize_signal(y, eps=0.01, axis=None)
            yield x, y

    def uniform_patient_generator(
        self,
        patient_ids: npt.NDArray,
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
            num_workers (int | None, optional): # parallel workers. Defaults to None.
            force (bool, optional): Force redownload. Defaults to False.
        """
        # Nothing to do
