import contextlib
import io
import logging
import os
import random
import uuid
from typing import Generator

import h5py
import numpy as np
import numpy.typing as npt
import physiokit as pk
from pydantic import BaseModel, Field

from .dataset import HKDataset
from .defines import PatientGenerator
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
    sample_rate: float = Field(500, description="Signal sample rate (Hz)")
    duration: int = Field(10, description="Signal duration in sec")
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
        num_pts: int = 250,
        leads: list[int] | None = None,
        params: dict | None = None,
    ) -> None:
        super().__init__(
            ds_path=ds_path,
        )
        self._noise_gen = None
        self._num_pts = num_pts
        self.leads = leads or list(range(12))
        self.params = SyntheticParams(**params or {})
        self._unique_id = str(uuid.uuid4())
        self._cache: dict[str, io.BytesIO] = {}
        os.makedirs(self.ds_path, exist_ok=True)

    @property
    def name(self) -> str:
        """Dataset name"""
        return "synthetic"

    @property
    def cachable(self) -> bool:
        """If dataset supports file caching."""
        return True

    @property
    def sampling_rate(self) -> int:
        """Sampling rate in Hz"""
        return self.params.sample_rate

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

    def pt_key(self, patient_id: int):
        """Get patient key"""
        return f"{patient_id:05d}"

    @contextlib.contextmanager
    def patient_data(self, patient_id: int) -> Generator[h5py.Group, None, None]:
        """Get patient data

        Args:
            patient_id (int): Patient ID

        Returns:
            Generator[h5py.Group, None, None]: Patient data
        """

        pt_key = self.pt_key(patient_id)
        if pt_key not in self._cache:
            ecg, segs, fids = self._synthesize_signal(
                frame_size=int(self.params.duration * self.sampling_rate), target_rate=self.sampling_rate
            )
            fp = io.BytesIO()
            with h5py.File(fp, mode="w") as h5:
                h5.create_dataset("data", data=ecg)
                h5.create_dataset("segmentations", data=segs)
                h5.create_dataset("fiducials", data=fids)
                h5.attrs["unique_id"] = self._unique_id
            # END WITH
            fp.seek(0)
            self._cache[pt_key] = fp
        # END IF

        with h5py.File(self._cache[pt_key], mode="r") as h5:
            yield h5
        # END WITH

    def signal_generator(
        self,
        patient_generator: PatientGenerator,
        frame_size: int,
        samples_per_patient: int = 1,
        target_rate: int | None = None,
    ) -> Generator[npt.NDArray, None, None]:
        """Generate frames using patient generator.

        Args:
            patient_generator (PatientGenerator): Generator that yields a tuple of patient id and patient data.
            samples_per_patient (int): Samples per patient.

        Returns:
            SampleGenerator: Generator of input data of shape (frame_size, 1)
        """
        if target_rate is None:
            target_rate = self.sampling_rate

        input_size = int(np.round((self.sampling_rate / target_rate) * frame_size))

        for pt in patient_generator:
            with self.patient_data(pt) as h5:
                data: h5py.Dataset = h5["data"][:]
            # END WITH
            for _ in range(samples_per_patient):
                lead = random.choice(self.leads)
                start = np.random.randint(0, data.shape[1] - input_size)
                x = data[lead, start : start + input_size].squeeze()
                x = np.nan_to_num(x).astype(np.float32)
                x = self.add_noise(x)
                if self.sampling_rate != target_rate:
                    x = pk.signal.resample_signal(x, self.sampling_rate, target_rate, axis=0)
                # END IF
                yield x
            # END FOR
        # END FOR

    def download(self, num_workers: int | None = None, force: bool = False):
        """Download dataset

        Args:
            num_workers (int | None, optional): # parallel workers. Defaults to None.
            force (bool, optional): Force redownload. Defaults to False.
        """
        # Nothing to do

    def close(self):
        """Close dataset"""
        if self._noise_gen is not None:
            self._noise_gen.close()
        # END IF
        self._cache.clear()

    def add_noise(self, ecg: npt.NDArray):
        """Add noise to ECG signal."""
        noise_range = self.params.noise_multiplier
        if noise_range[0] == 0 and noise_range[1] == 0:
            return ecg
        noise_level = np.random.uniform(noise_range[0], noise_range[1])

        if self._noise_gen is None:
            self._noise_gen = NstdbNoise(target_rate=self.sampling_rate)
        # END IF
        self._noise_gen.apply_noise(ecg, noise_level)
        return ecg

    def _synthesize_signal(
        self,
        frame_size: int,
        target_rate: float | None = None,
    ) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
        """Generate synthetic signal of given length

        Args:
            frame_size (int): Frame size

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
            signal_length=frame_size,
            sample_rate=target_rate,
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
