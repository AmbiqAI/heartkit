import logging
import os
import random

import numpy as np
import numpy.typing as npt
import physiokit as pk
import tensorflow as tf
from pydantic import BaseModel, Field

from ..tasks import HKSegment
from .dataset import HKDataset
from .defines import PatientGenerator, SampleGenerator
from .nstdb import NstdbNoise

logger = logging.getLogger(__name__)


class SyntheticParams(BaseModel, extra="allow"):
    """Synthetic parameters"""

    presets: list[pk.ecg.EcgPreset] = Field(
        default_factory=lambda: [
            pk.ecg.EcgPreset.SR,
            pk.ecg.EcgPreset.AFIB,
            pk.ecg.EcgPreset.LAHB,
            pk.ecg.EcgPreset.LPHB,
            pk.ecg.EcgPreset.LBBB,
            pk.ecg.EcgPreset.ant_STEMI,
            pk.ecg.EcgPreset.random_morphology,
            pk.ecg.EcgPreset.high_take_off,
        ],
        description="ECG presets",
    )
    preset_weights: list[int] = Field(
        default_factory=lambda: [14, 1, 1, 1, 1, 1, 1, 1], description="ECG preset weights"
    )
    heart_rate: tuple[float, float] = Field((40, 120), description="Heart rate range")
    impedance: tuple[float, float] = Field((1.0, 2.0), description="Impedance range")
    p_multiplier: tuple[float, float] = Field((0.80, 1.2), description="P wave width multiplier range")
    t_multiplier: tuple[float, float] = Field((0.80, 1.2), description="T wave width multiplier range")
    noise_multiplier: tuple[float, float] = Field((0, 0), description="Noise multiplier range")
    voltage_factor: tuple[float, float] = Field((800, 1000), description="Voltage factor range")


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
        noise_level: float = 0.0,
        leads: list[int] | None = None,
        params: dict | None = None,
    ) -> None:
        super().__init__(
            ds_path=ds_path / "synthetic",
            task=task,
            frame_size=frame_size,
            target_rate=target_rate,
            spec=spec,
            class_map=class_map,
        )
        self._noise_gen = None
        self._num_pts = num_pts
        self.noise_level = noise_level
        self.leads = leads or list(range(12))
        self.params = SyntheticParams(**params or {})

    @property
    def cachable(self) -> bool:
        """If dataset supports file caching."""
        return True

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

    def _synthesize_signal(self, signal_length: int) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
        """Generate synthetic signal of given length

        Args:
            signal_length (int): Signal length

        Returns:
            tuple[npt.NDArray, npt.NDArray, npt.NDArray]: signal, segments, fiducials
        """
        heart_rate = np.random.uniform(self.params.heart_rate[0], self.params.heart_rate[1])
        preset = random.choices(self.params.presets, self.params.preset_weights, k=1)[0].value
        impedance = np.random.uniform(self.params.impedance[0], self.params.impedance[1])
        p_multiplier = np.random.uniform(self.params.p_multiplier[0], self.params.p_multiplier[1])
        t_multiplier = np.random.uniform(self.params.t_multiplier[0], self.params.t_multiplier[1])
        voltage_factor = np.random.uniform(self.params.voltage_factor[0], self.params.voltage_factor[1])

        ecg, segs, fids = pk.ecg.synthesize(
            signal_length=signal_length,
            sample_rate=self.sampling_rate,
            leads=12,  # Use all 12 leads
            heart_rate=heart_rate,
            preset=preset,
            impedance=impedance,
            p_multiplier=p_multiplier,
            t_multiplier=t_multiplier,
            noise_multiplier=0,
            voltage_factor=voltage_factor,
        )
        return ecg, segs, fids

    def signal_generator(self, patient_generator: PatientGenerator, samples_per_patient: int = 1) -> SampleGenerator:
        """Generate frames using patient generator.

        Args:
            patient_generator (PatientGenerator): Generator that yields a tuple of patient id and patient data.
            samples_per_patient (int): Samples per patient.

        Returns:
            SampleGenerator: Generator of input data of shape (frame_size, 1)
        """

        signal_length = max(2 * self.frame_size, int(self.frame_size * samples_per_patient / len(self.leads)))
        for _ in patient_generator:
            syn_ecg, _, _ = self._synthesize_signal(signal_length)
            for _ in range(samples_per_patient):
                lead = random.choice(self.leads)
                frame_start = np.random.randint(0, syn_ecg.shape[1] - self.frame_size)
                frame_end = frame_start + self.frame_size
                x = syn_ecg[lead, frame_start:frame_end].astype(np.float32).reshape((self.frame_size,))
                x = self._add_noise(x)
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
        signal_length = max(2 * self.frame_size, int(self.frame_size * samples_per_patient / len(self.leads)))

        for _ in patient_generator:
            syn_ecg, syn_segs_t, _ = self._synthesize_signal(signal_length)
            syn_segs = np.zeros_like(syn_segs_t)
            for i in range(syn_segs_t.shape[0]):
                syn_segs[i, np.where((syn_segs_t[i] == pk.ecg.EcgSegment.tp_overlap))[0]] = HKSegment.pwave
                syn_segs[i, np.where((syn_segs_t[i] == pk.ecg.EcgSegment.p_wave))[0]] = HKSegment.pwave
                syn_segs[i, np.where((syn_segs_t[i] == pk.ecg.EcgSegment.qrs_complex))[0]] = HKSegment.qrs
                syn_segs[i, np.where((syn_segs_t[i] == pk.ecg.EcgSegment.t_wave))[0]] = HKSegment.twave
            # END FOR

            for i in range(samples_per_patient):
                lead = random.choice(self.leads)
                frame_start = np.random.randint(start_offset, syn_ecg.shape[1] - self.frame_size)
                frame_end = frame_start + self.frame_size
                x = syn_ecg[lead, frame_start:frame_end].astype(np.float32)
                x = self._add_noise(x)
                y = syn_segs[lead, frame_start:frame_end].astype(np.int32)
                y = np.vectorize(self.class_map.get, otypes=[int])(y)
                yield x, y
            # END FOR
        # END FOR

    def denoising_generator(
        self,
        patient_generator: PatientGenerator,
        samples_per_patient: int | list[int] = 1,
    ) -> SampleGenerator:
        """Generate frames and noise frames."""
        signal_length = max(2 * self.frame_size, int(self.frame_size * samples_per_patient / len(self.leads)))
        for _ in patient_generator:
            syn_ecg, _, _ = self._synthesize_signal(signal_length)
            for _ in range(samples_per_patient):
                lead = random.choice(self.leads)
                frame_start = np.random.randint(0, syn_ecg.shape[1] - self.frame_size)
                frame_end = frame_start + self.frame_size
                x = syn_ecg[lead, frame_start:frame_end].astype(np.float32).reshape((self.frame_size,))
                y = x.copy()
                x = self._add_noise(x)
                yield x, y
            # END FOR
        # END FOR

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

    def _add_noise(self, ecg: npt.NDArray):
        """Add noise to ECG signal."""
        noise_range = self.params.noise_multiplier
        if noise_range[0] == 0 and noise_range[1] == 0:
            return ecg
        noise_level = np.random.uniform(noise_range[0], noise_range[1])

        if self._noise_gen is None:
            self._noise_gen = NstdbNoise(ds_path=self.ds_path.parent, target_rate=self.target_rate)
        # END IF
        self._noise_gen.apply_noise(ecg, noise_level)
        return ecg
