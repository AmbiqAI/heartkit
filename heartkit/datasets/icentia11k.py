import functools
import logging
import os
import random
import tempfile
import zipfile
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from enum import IntEnum
from multiprocessing import Pool

import boto3
import h5py
import numpy as np
import numpy.typing as npt
import pandas as pd
import physiokit as pk
import sklearn.model_selection
import sklearn.preprocessing
import tensorflow as tf
from botocore import UNSIGNED
from botocore.client import Config
from tqdm import tqdm

from ..tasks import HeartRate, HKBeat, HKRhythm, HKSegment
from ..utils import download_file
from .dataset import HKDataset
from .defines import PatientGenerator, SampleGenerator

logger = logging.getLogger(__name__)


class IcentiaRhythm(IntEnum):
    """Icentia rhythm labels"""

    noise = 0
    normal = 1
    afib = 2
    aflut = 3
    end = 4


class IcentiaBeat(IntEnum):
    """Incentia beat labels"""

    undefined = 0
    normal = 1
    pac = 2
    aberrated = 3
    pvc = 4


# These map Icentia specific labels to common labels
IcentiaRhythmMap = {
    IcentiaRhythm.noise: HKRhythm.noise,
    IcentiaRhythm.normal: HKRhythm.sr,
    IcentiaRhythm.afib: HKRhythm.afib,
    IcentiaRhythm.aflut: HKRhythm.aflut,
    IcentiaRhythm.end: HKRhythm.noise,
}

IcentiaBeatMap = {
    IcentiaBeat.undefined: HKBeat.noise,
    IcentiaBeat.normal: HKBeat.normal,
    IcentiaBeat.pac: HKBeat.pac,
    IcentiaBeat.aberrated: HKBeat.pac,
    IcentiaBeat.pvc: HKBeat.pvc,
}

IcentiaLeadsMap = {
    "i": 0,  # Modified lead I
}


class IcentiaDataset(HKDataset):
    """Icentia dataset"""

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
            ds_path=ds_path / "icentia11k",
            task=task,
            frame_size=frame_size,
            target_rate=target_rate,
            spec=spec,
            class_map=class_map,
        )

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
        if self.task == "rhythm":
            return self.rhythm_data_generator(
                patient_generator=patient_generator,
                samples_per_patient=samples_per_patient,
            )
        if self.task == "beat":
            return self.beat_data_generator(
                patient_generator=patient_generator,
                samples_per_patient=samples_per_patient,
            )
        if self.task == "segmentation":
            return self.segmentation_generator(
                patient_generator=patient_generator,
                samples_per_patient=samples_per_patient,
            )
        raise NotImplementedError()

    def rhythm_data_generator(
        self,
        patient_generator: PatientGenerator,
        samples_per_patient: int | list[int] = 1,
    ) -> SampleGenerator:
        """Generate frames w/ rhythm labels (e.g. afib) using patient generator.

        Args:
            patient_generator (PatientGenerator): Patient Generator
            samples_per_patient (int | list[int], optional): # samples per patient. Defaults to 1.

        Returns:
            SampleGenerator: Sample generator

        Yields:
            Iterator[SampleGenerator]
        """
        # Target labels and mapping
        tgt_labels = list(set(self.class_map.values()))

        # Convert local labels -> HK labels -> class map labels (-1 indicates not in class map)
        tgt_map = {k: self.class_map.get(v, -1) for (k, v) in IcentiaRhythmMap.items()}
        num_classes = len(tgt_labels)

        # If samples_per_patient is a list, then it must be the same length as num_classes
        if isinstance(samples_per_patient, Iterable):
            samples_per_tgt = samples_per_patient
        else:
            num_per_tgt = int(max(1, samples_per_patient / num_classes))
            samples_per_tgt = num_per_tgt * [num_classes]

        input_size = int(np.round((self.sampling_rate / self.target_rate) * self.frame_size))

        # Group patient rhythms by type (segment, start, stop, delta)
        for _, segments in patient_generator:
            # This maps segment index to segment key
            seg_map: list[str] = list(segments.keys())

            pt_tgt_seg_map = [[] for _ in tgt_labels]
            for seg_idx, seg_key in enumerate(seg_map):
                # Grab rhythm labels
                rlabels = segments[seg_key]["rlabels"][:]

                # Skip if no rhythm labels
                if not rlabels.shape[0]:
                    continue
                rlabels = rlabels[np.where(rlabels[:, 1] != IcentiaRhythm.noise.value)[0]]
                if not rlabels.shape[0]:
                    continue

                # Unpack start, end, and label
                xs, xe, xl = rlabels[0::2, 0], rlabels[1::2, 0], rlabels[0::2, 1]

                # Map labels to target labels
                xl = np.vectorize(tgt_map.get, otypes=[int])(xl)

                # Capture segment, start, and end for each target label
                for tgt_idx, tgt_class in enumerate(tgt_labels):
                    idxs = np.where((xe - xs >= input_size) & (xl == tgt_class))
                    seg_vals = np.vstack((seg_idx * np.ones_like(idxs), xs[idxs], xe[idxs])).T
                    pt_tgt_seg_map[tgt_idx] += seg_vals.tolist()
                # END FOR
            # END FOR
            pt_tgt_seg_map = [np.array(b) for b in pt_tgt_seg_map]

            # Grab target segments
            seg_samples: list[tuple[int, int, int, int]] = []
            for tgt_idx, tgt_class in enumerate(tgt_labels):
                tgt_segments = pt_tgt_seg_map[tgt_idx]
                if not tgt_segments.shape[0]:
                    continue
                tgt_seg_indices: list[int] = random.choices(
                    np.arange(tgt_segments.shape[0]),
                    weights=tgt_segments[:, 2] - tgt_segments[:, 1],
                    k=samples_per_tgt[tgt_idx],
                )
                for tgt_seg_idx in tgt_seg_indices:
                    seg_idx, rhy_start, rhy_end = tgt_segments[tgt_seg_idx]
                    frame_start = np.random.randint(rhy_start, rhy_end - input_size + 1)
                    frame_end = frame_start + input_size
                    seg_samples.append((seg_idx, frame_start, frame_end, tgt_class))
                # END FOR
            # END FOR

            # Shuffle segments
            random.shuffle(seg_samples)

            # Yield selected samples for patient
            for seg_idx, frame_start, frame_end, label in seg_samples:
                x: npt.NDArray = segments[seg_map[seg_idx]]["data"][frame_start:frame_end].astype(np.float32)
                if self.sampling_rate != self.target_rate:
                    x = pk.signal.resample_signal(x, self.sampling_rate, self.target_rate, axis=0)
                yield x, label
            # END FOR
        # END FOR

    def segmentation_generator(
        self,
        patient_generator: PatientGenerator,
        samples_per_patient: int | list[int] = 1,
    ) -> SampleGenerator:
        """Generate frames with annotated segments.

        Args:
            patient_generator (PatientGenerator): Patient generator
            samples_per_patient (int | list[int], optional):

        Returns:
            SampleGenerator: Sample generator
        """
        assert not isinstance(samples_per_patient, Iterable)
        input_size = int(np.round((self.sampling_rate / self.target_rate) * self.frame_size))

        # For each patient
        for _, segments in patient_generator:
            for _ in range(samples_per_patient):
                # Randomly pick a segment
                seg_key = np.random.choice(list(segments.keys()))
                # Randomly pick a frame
                frame_start = np.random.randint(segments[seg_key]["data"].shape[0] - input_size)
                frame_end = frame_start + input_size
                # Get data and labels
                data = segments[seg_key]["data"][frame_start:frame_end].squeeze()

                if self.sampling_rate != self.target_rate:
                    ds_ratio = self.target_rate / self.sampling_rate
                    data = pk.signal.resample_signal(data, self.sampling_rate, self.target_rate, axis=0)
                else:
                    ds_ratio = 1

                blabels = segments[seg_key]["blabels"]
                blabels = blabels[(blabels[:, 0] >= frame_start) & (blabels[:, 0] < frame_end)]
                # Create segment mask
                mask = np.zeros_like(data, dtype=np.int32)

                # Check if pwave, twave, or uwave are in class_map- if so, add gradient filter to mask
                non_qrs = [self.class_map.get(k, -1) for k in (HKSegment.pwave, HKSegment.twave, HKSegment.uwave)]
                if any((v != -1 for v in non_qrs)):
                    xc = pk.ecg.clean(data.copy(), sample_rate=self.target_rate, lowcut=0.5, highcut=40, order=3)
                    grad = pk.signal.moving_gradient_filter(
                        xc, sample_rate=self.target_rate, sig_window=0.1, avg_window=1.0, sig_prom_weight=0.15
                    )
                    mask[grad > 0] = -1
                # END IF

                for i in range(blabels.shape[0]):
                    bidx = int((blabels[i, 0] - frame_start) * ds_ratio)
                    btype = blabels[i, 1]

                    # Unclassifiable beat (treat as noise?)
                    if btype == IcentiaBeat.undefined:
                        pass
                        # noise_lbl = self.class_map.get(HeartSegment.noise.value, -1)
                        # # Skip if not in class map
                        # if noise_lbl == -1
                        #     continue
                        # # Mark region as noise
                        # win_len = max(1, int(0.2 * self.target_rate))  # 200 ms
                        # b_left = max(0, bidx - win_len)
                        # b_right = min(data.shape[0], bidx + win_len)
                        # mask[b_left:b_right] = noise_lbl

                    # Normal, PAC, PVC beat
                    else:
                        qrs_width = int(0.08 * self.target_rate)  # 80 ms
                        # Extract QRS segment
                        qrs = pk.signal.moving_gradient_filter(
                            data, sample_rate=self.target_rate, sig_window=0.1, avg_window=1.0, sig_prom_weight=1.5
                        )
                        win_len = max(1, qrs_width)
                        b_left = max(0, bidx - win_len)
                        b_right = min(data.shape[0], bidx + win_len)
                        onset = np.where(np.flip(qrs[b_left:bidx]) < 0)[0]
                        onset = onset[0] if onset.size else win_len
                        offset = np.where(qrs[bidx + 1 : b_right] < 0)[0]
                        offset = offset[0] if offset.size else win_len
                        qrs_onset = bidx - onset
                        qrs_offset = bidx + offset
                        mask[qrs_onset:qrs_offset] = self.class_map.get(HKSegment.qrs.value, 0)
                    # END IF
                # END FOR
                x = np.nan_to_num(data).astype(np.float32)
                y = mask.astype(np.int32)
                yield x, y
            # END FOR
        # END FOR

    def beat_data_generator(
        self,
        patient_generator: PatientGenerator,
        samples_per_patient: int | list[int] = 1,
    ) -> SampleGenerator:
        """Generate frames and beat label using patient generator.

        Args:
            patient_generator (PatientGenerator): Patient generator
            samples_per_patient (int | list[int], optional): # samples per patient. Defaults to 1.

        Returns:
            SampleGenerator: Sample generator

        Yields:
            Iterator[SampleGenerator]
        """
        nlabel_threshold = 0.25
        blabel_padding = 20
        rr_win_len = int(10 * self.sampling_rate)
        rr_min_len = int(0.3 * self.sampling_rate)
        rr_max_len = int(2.0 * self.sampling_rate)

        # Target labels and mapping
        num_classes = len(set(self.class_map.values()))

        # Convert Icentia labels -> HK labels -> class map labels (-1 indicates not in class map)
        tgt_map = {k: self.class_map.get(v, -1) for (k, v) in IcentiaBeatMap.items()}

        # If samples_per_patient is a list, then it must be the same length as nclasses
        if isinstance(samples_per_patient, Iterable):
            samples_per_tgt = samples_per_patient
        else:
            num_per_tgt = int(max(1, samples_per_patient / num_classes))
            samples_per_tgt = num_per_tgt * [num_classes]

        input_size = int(np.round((self.sampling_rate / self.target_rate) * self.frame_size))

        # Filter beats based on neighboring beats
        def filter_func(blabels: npt.NDArray, beat: IcentiaBeat, i: int):
            match beat:
                case IcentiaBeat.normal:
                    return blabels[i - 1, 1] == blabels[i + 1, 1] == IcentiaBeat.normal
                case IcentiaBeat.pac, IcentiaBeat.pvc:
                    return IcentiaBeat.undefined not in (
                        blabels[i - 1, 1],
                        blabels[i + 1, 1],
                    )
                case IcentiaBeat.undefined:
                    return blabels[i - 1, 1] == blabels[i + 1, 1] == IcentiaBeat.undefined
                case _:
                    return True
            # END MATCH

        # END DEF

        # For each patient
        for _, segments in patient_generator:
            # This maps segment index to segment key
            seg_map: list[str] = list(segments.keys())

            # Capture beat locations for each segment
            pt_beat_map = [[] for _ in range(num_classes)]
            for seg_idx, seg_key in enumerate(seg_map):
                # Get beat labels
                blabels = segments[seg_key]["blabels"][:]

                # If no beats, skip
                num_blabels = blabels.shape[0]
                if num_blabels <= 0:
                    continue
                # END IF

                # If too few normal beats, skip
                num_nlabels = np.sum(blabels[:, 1] == IcentiaBeat.normal)
                if num_nlabels / num_blabels < nlabel_threshold:
                    continue

                # Capture all beat locations
                for beat in IcentiaBeat:
                    # Skip if not in class map
                    beat_class = tgt_map.get(beat, -1)
                    if beat_class < 0 or beat_class >= num_classes:
                        continue

                    # Get all beat type indices
                    beat_idxs = np.where(blabels[blabel_padding:-blabel_padding, 1] == beat.value)[0] + blabel_padding

                    # Filter indices
                    beat_idxs = filter(functools.partial(filter_func, blabels, beat), beat_idxs)
                    pt_beat_map[beat_class] += [(seg_idx, blabels[i, 0]) for i in beat_idxs]
                # END FOR
            # END FOR
            pt_beat_map = [np.array(b) for b in pt_beat_map]

            # Randomly select N samples of each target beat
            pt_segs_beat_idxs: list[tuple[int, int, int]] = []
            for tgt_beat_idx, tgt_beats in enumerate(pt_beat_map):
                tgt_count = min(samples_per_tgt[tgt_beat_idx], len(tgt_beats))
                tgt_idxs = np.random.choice(np.arange(len(tgt_beats)), size=tgt_count, replace=False)
                pt_segs_beat_idxs += [(tgt_beats[i][0], tgt_beats[i][1], tgt_beat_idx) for i in tgt_idxs]
            # END FOR

            # Shuffle all
            random.shuffle(pt_segs_beat_idxs)

            # Yield selected samples for patient
            for seg_idx, beat_idx, beat in pt_segs_beat_idxs:
                frame_start = max(0, beat_idx - int(random.uniform(0.4722, 0.5278) * input_size))
                frame_end = frame_start + input_size
                data = segments[seg_map[seg_idx]]["data"]
                blabels = segments[seg_map[seg_idx]]["blabels"]

                # Compute average RR interval
                rr_xs = np.searchsorted(blabels[:, 0], max(0, frame_start - rr_win_len))
                rr_xe = np.searchsorted(blabels[:, 0], frame_end + rr_win_len)
                if rr_xe <= rr_xs:
                    continue
                rri = np.diff(blabels[rr_xs : rr_xe + 1, 0])
                rri = rri[(rri > rr_min_len) & (rri < rr_max_len)]
                if rri.size <= 0:
                    continue
                avg_rr = int(np.mean(rri))

                if frame_start - avg_rr < 0 or frame_end + avg_rr >= data.shape[0]:
                    continue

                # Combine previous, current, and next beat
                x = np.hstack(
                    (
                        data[frame_start - avg_rr : frame_end - avg_rr],
                        data[frame_start:frame_end],
                        data[frame_start + avg_rr : frame_end + avg_rr],
                    )
                )
                x = np.nan_to_num(x).astype(np.float32)
                if self.sampling_rate != self.target_rate:
                    x = pk.signal.resample_signal(x, self.sampling_rate, self.target_rate, axis=0)
                y = beat
                yield x, y
            # END FOR
        # END FOR

    def heart_rate_data_generator(
        self,
        patient_generator: PatientGenerator,
        samples_per_patient: int = 1,
    ) -> SampleGenerator:
        """Generate frames and heart rate label using patient generator.

        Args:
            patient_generator (PatientGenerator): Patient generator
            samples_per_patient (int, optional): # samples per patient. Defaults to 1.

        Returns:
            SampleGenerator: Sample generator

        Yields:
            Iterator[SampleGenerator]
        """

        label_frame_size = self.frame_size
        max_frame_size = max(self.frame_size, label_frame_size)
        for _, segments in patient_generator:
            for _ in range(samples_per_patient):
                segment = segments[np.random.choice(list(segments.keys()))]
                segment_size: int = segment["data"].shape[0]
                frame_center = np.random.randint(segment_size - max_frame_size) + max_frame_size // 2
                signal_frame_start = frame_center - self.frame_size // 2
                signal_frame_end = frame_center + self.frame_size // 2
                x = segment["data"][signal_frame_start:signal_frame_end]
                if self.sampling_rate != self.target_rate:
                    x = pk.signal.resample_signal(x, self.sampling_rate, self.target_rate, axis=0)
                label_frame_start = frame_center - label_frame_size // 2
                label_frame_end = frame_center + label_frame_size // 2
                beat_indices = segment["blabels"][:, 0]
                frame_beat_indices = self.get_complete_beats(beat_indices, start=label_frame_start, end=label_frame_end)
                y = self._get_heart_rate_label(frame_beat_indices, self.sampling_rate)
                yield x, y
            # END FOR
        # END FOR

    def signal_generator(self, patient_generator: PatientGenerator, samples_per_patient: int = 1) -> SampleGenerator:
        """Generate random frames using patient generator.

        Args:
            patient_generator (PatientGenerator): Generator that yields a tuple of patient id and patient data.
                    Patient data may contain only signals, since labels are not used.
            samples_per_patient (int): Samples per patient.

        Returns:
            SampleGenerator: Generator of input data of shape (frame_size, 1)
        """
        input_size = int(np.round((self.sampling_rate / self.target_rate) * self.frame_size))
        for _, segments in patient_generator:
            for _ in range(samples_per_patient):
                segment = segments[np.random.choice(list(segments.keys()))]
                segment_size = segment["data"].shape[0]
                frame_start = np.random.randint(segment_size - input_size)
                frame_end = frame_start + input_size
                x = segment["data"][frame_start:frame_end].squeeze()
                x = np.nan_to_num(x).astype(np.float32)
                if self.sampling_rate != self.target_rate:
                    x = pk.signal.resample_signal(x, self.sampling_rate, self.target_rate, axis=0)
                # END IF
                yield x
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
                pt_key = self._pt_key(patient_id)
                with h5py.File(self.ds_path / f"{pt_key}.h5", mode="r") as h5:
                    patient_data = h5[pt_key]
                    yield patient_id, patient_data
                # END WITH
            # END FOR
            if not repeat:
                break
            # END IF
        # END WHILE

    def random_patient_generator(
        self,
        patient_ids: list[int],
        patient_weights: list[int] | None = None,
    ) -> PatientGenerator:
        """Samples patient data from the provided patient distribution.

        Args:
            patient_ids (list[int]): Patient ids
            patient_weights (list[int] | None, optional): Probabilities associated with each patient. Defaults to None.

        Returns:
            PatientGenerator: Patient generator

        Yields:
            Iterator[PatientGenerator]
        """
        while True:
            for patient_id in np.random.choice(patient_ids, size=1024, p=patient_weights):
                pt_key = self._pt_key(patient_id)
                with h5py.File(self.ds_path / f"{pt_key}.h5", mode="r") as h5:
                    patient_data = h5[pt_key]
                    yield patient_id, patient_data
                # END WITH
            # END FOR
        # END WHILE

    def get_complete_beats(
        self,
        indices: npt.NDArray,
        labels: npt.NDArray | None = None,
        start: int = 0,
        end: int | None = None,
    ) -> tuple[npt.NDArray, npt.NDArray]:
        """Find all complete beats within a frame i.e. start and end of the beat lie within the frame.
        The indices are assumed to specify the end of a heartbeat.

        Args:
            indices (npt.NDArray): List of sorted beat indices.
            labels (npt.NDArray | None): List of beat labels. Defaults to None.
            start (int): Index of the first sample in the frame. Defaults to 0.
            end (int | None): Index of the last sample in the frame. Defaults to None.

        Returns:
            tuple[npt.NDArray, npt.NDArray]: (beat indices, beat labels)
        """
        if end is None:
            end = indices[-1]
        if start >= end:
            raise ValueError("`end` must be greater than `start`")
        start_index = np.searchsorted(indices, start, side="left") + 1
        end_index = np.searchsorted(indices, end, side="right")
        indices_slice = indices[start_index:end_index]
        if labels is None:
            return indices_slice
        label_slice = labels[start_index:end_index]
        return (indices_slice, label_slice)

    def _get_heart_rate_label(self, qrs_indices, fs=None) -> int:
        """Determine the heart rate label based on an array of QRS indices (separating individual heartbeats).
            The QRS indices are assumed to be measured in seconds if sampling frequency `fs` is not specified.
            The heartbeat label is based on the following BPM (beats per minute) values: (0) tachycardia <60 BPM,
            (1) bradycardia >100 BPM, (2) healthy 60-100 BPM, (3) noisy if QRS detection failed.

        Args:
            qrs_indices (list[int]): Array of QRS indices.
            fs (float, optional): Sampling frequency of the signal. Defaults to None.

        Returns:
            int: Heart rate label
        """
        if not qrs_indices:
            return HeartRate.noise.value

        rr_intervals = np.diff(qrs_indices)
        if fs is not None:
            rr_intervals = rr_intervals / fs
        bpm = 60 / rr_intervals.mean()
        if bpm < 60:
            return HeartRate.bradycardia.value
        if bpm <= 100:
            return HeartRate.sinus.value
        return HeartRate.tachycardia.value

    def get_rhythm_statistics(
        self,
        patient_ids: npt.NDArray | None = None,
        save_path: str | None = None,
    ) -> pd.DataFrame:
        """Utility function to extract rhythm statistics across entire dataset. Useful for EDA.

        Args:
            patient_ids (npt.NDArray | None, optional): Patients IDs to include. Defaults to all.
            save_path (str | None, optional): Parquet file path to save results. Defaults to None.

        Returns:
            pd.DataFrame: DataFrame of statistics
        """

        if patient_ids is None:
            patient_ids = self.patient_ids
        pt_gen = self.uniform_patient_generator(patient_ids=patient_ids, repeat=False)
        stats = []
        for pt, segments in pt_gen:
            # Group patient rhythms by type (segment, start, stop)
            segment_label_map: dict[str, list[tuple[str, int, int]]] = {}
            for seg_key, segment in segments.items():
                rlabels = segment["rlabels"][:]
                if rlabels.shape[0] == 0:
                    continue  # Segment has no rhythm labels
                rlabels = rlabels[np.where(rlabels[:, 1] != IcentiaRhythm.noise.value)[0]]
                for i, l in enumerate(rlabels[::2, 1]):
                    if l in (
                        IcentiaRhythm.noise,
                        IcentiaRhythm.normal,
                        IcentiaRhythm.afib,
                        IcentiaRhythm.aflut,
                    ):
                        rhy_start, rhy_stop = (
                            rlabels[i * 2 + 0, 0],
                            rlabels[i * 2 + 1, 0],
                        )
                        stats.append(
                            dict(
                                pt=pt,
                                rc=seg_key,
                                rhythm=l,
                                start=rhy_start,
                                stop=rhy_stop,
                                dur=rhy_stop - rhy_start,
                            )
                        )
                        segment_label_map[l] = segment_label_map.get(l, []) + [
                            (seg_key, rlabels[i * 2 + 0, 0], rlabels[i * 2 + 1, 0])
                        ]
                    # END IF
                # END FOR
            # END FOR
        # END FOR
        df = pd.DataFrame(stats)
        if save_path:
            df.to_parquet(save_path)
        return df

    def download(self, num_workers: int | None = None, force: bool = False):
        """Download dataset

        This will download preprocessed HDF5 files from S3.

        Args:
            num_workers (int | None, optional): # parallel workers. Defaults to None.
            force (bool, optional): Force redownload. Defaults to False.
        """

        def download_s3_file(
            s3_file: str,
            save_path: os.PathLike,
            bucket: str,
            client: boto3.client,
            force: bool = False,
        ):
            if not force and os.path.exists(save_path):
                return
            client.download_file(
                Bucket=bucket,
                Key=s3_file,
                Filename=str(save_path),
            )

        s3_bucket = "ambiqai-ecg-icentia11k-dataset"
        s3_prefix = "patients"

        os.makedirs(self.ds_path, exist_ok=True)

        patient_ids = self.patient_ids

        # Creating only one session and one client
        session = boto3.Session()
        client = session.client("s3", config=Config(signature_version=UNSIGNED))

        func = functools.partial(download_s3_file, bucket=s3_bucket, client=client, force=force)

        with tqdm(desc="Downloading icentia11k dataset from S3", total=len(patient_ids)) as pbar:
            pt_keys = [self._pt_key(patient_id) for patient_id in patient_ids]
            with ThreadPoolExecutor(max_workers=2 * num_workers) as executor:
                futures = (
                    executor.submit(
                        func,
                        f"{s3_prefix}/{pt_key}.h5",
                        self.ds_path / f"{pt_key}.h5",
                    )
                    for pt_key in pt_keys
                )
                for future in as_completed(futures):
                    err = future.exception()
                    if err:
                        logger.exception("Failed on file")
                    pbar.update(1)
                # END FOR
            # END WITH
        # END WITH

    def download_raw_dataset(self, num_workers: int | None = None, force: bool = False):
        """Downloads full Icentia dataset zipfile and converts into individial patient HDF5 files.
        NOTE: This is a very long process (e.g. 24 hrs). Please use `icentia11k.download_dataset` instead.

        Args:
            force (bool, optional): Whether to force re-download if destination exists. Defaults to False.
            num_workers (int, optional): # parallel workers. Defaults to os.cpu_count().
        """
        logger.info("Downloading icentia11k dataset")
        ds_url = (
            "https://physionet.org/static/published-projects/icentia11k-continuous-ecg/"
            "icentia11k-single-lead-continuous-raw-electrocardiogram-dataset-1.0.zip"
        )
        ds_zip_path = self.ds_path / "icentia11k.zip"
        os.makedirs(self.ds_path, exist_ok=True)
        if os.path.exists(ds_zip_path) and not force:
            logger.warning(
                f"Zip file already exists. Please delete or set `force` flag to redownload. PATH={ds_zip_path}"
            )
        else:
            download_file(ds_url, ds_zip_path, progress=True)

        # 2. Extract and convert patient ECG data to H5 files
        logger.info("Generating icentia11k patient data")
        self._convert_dataset_zip_to_hdf5(
            zip_path=ds_zip_path,
            force=force,
            num_workers=num_workers,
        )
        logger.info("Finished icentia11k patient data")

    def _convert_dataset_pt_zip_to_hdf5(self, patient: int, zip_path: os.PathLike, force: bool = False):
        """Extract patient data from Icentia zipfile. Pulls out ECG data along with all labels.

        Args:
            patient (int): Patient id
            zip_path (PathLike): Zipfile path
            force (bool, optional): Whether to override destination if it exists. Defaults to False.
        """
        import re  # pylint: disable=import-outside-toplevel

        import wfdb  # pylint: disable=import-outside-toplevel

        # These map Wfdb labels to icentia labels
        WfdbRhythmMap = {
            "": IcentiaRhythm.noise.value,
            "(N": IcentiaRhythm.normal.value,
            "(AFIB": IcentiaRhythm.afib.value,
            "(AFL": IcentiaRhythm.aflut.value,
            ")": IcentiaRhythm.end.value,
        }
        WfdbBeatMap = {
            "Q": IcentiaBeat.undefined.value,
            "N": IcentiaBeat.normal.value,
            "S": IcentiaBeat.pac.value,
            "a": IcentiaBeat.aberrated.value,
            "V": IcentiaBeat.pvc.value,
        }

        logger.info(f"Processing patient {patient}")
        pt_id = self._pt_key(patient)
        pt_path = self.ds_path / f"{pt_id}.h5"
        if not force and os.path.exists(pt_path):
            logger.debug(f"Skipping patient {pt_id}")
            return
        zp = zipfile.ZipFile(zip_path, mode="r")  # pylint: disable=consider-using-with
        h5 = h5py.File(pt_path, mode="w")

        # Find all patient .dat file indices
        zp_rec_names = filter(
            lambda f: re.match(f"{pt_id}_[A-z0-9]+.dat", os.path.basename(f)),
            (f.filename for f in zp.filelist),
        )
        for zp_rec_name in zp_rec_names:
            try:
                zp_hdr_name = zp_rec_name.replace(".dat", ".hea")
                zp_atr_name = zp_rec_name.replace(".dat", ".atr")

                with tempfile.TemporaryDirectory() as tmpdir:
                    rec_fpath = os.path.join(tmpdir, os.path.basename(zp_rec_name))
                    atr_fpath = rec_fpath.replace(".dat", ".atr")
                    hdr_fpath = rec_fpath.replace(".dat", ".hea")
                    with open(hdr_fpath, "wb") as fp:
                        fp.write(zp.read(zp_hdr_name))
                    with open(rec_fpath, "wb") as fp:
                        fp.write(zp.read(zp_rec_name))
                    with open(atr_fpath, "wb") as fp:
                        fp.write(zp.read(zp_atr_name))
                    rec = wfdb.rdrecord(os.path.splitext(rec_fpath)[0], physical=True)
                    atr = wfdb.rdann(os.path.splitext(atr_fpath)[0], extension="atr")
                pt_seg_path = f"/{os.path.splitext(os.path.basename(zp_rec_name))[0].replace('_', '/')}"
                data = rec.p_signal.astype(np.float16)
                blabels = np.array(
                    [[atr.sample[i], WfdbBeatMap.get(s)] for i, s in enumerate(atr.symbol) if s in WfdbBeatMap],
                    dtype=np.int32,
                )
                rlabels = np.array(
                    [
                        [atr.sample[i], WfdbRhythmMap.get(atr.aux_note[i], 0)]
                        for i, s in enumerate(atr.symbol)
                        if s == "+"
                    ],
                    dtype=np.int32,
                )
                h5.create_dataset(
                    name=f"{pt_seg_path}/data",
                    data=data,
                    compression="gzip",
                    compression_opts=3,
                )
                h5.create_dataset(name=f"{pt_seg_path}/blabels", data=blabels)
                h5.create_dataset(name=f"{pt_seg_path}/rlabels", data=rlabels)
            except Exception as err:  # pylint: disable=broad-except
                logger.warning(f"Failed processing {zp_rec_name}", err)
                continue
        h5.close()

    def _convert_dataset_zip_to_hdf5(
        self,
        zip_path: os.PathLike,
        patient_ids: npt.NDArray | None = None,
        force: bool = False,
        num_workers: int | None = None,
    ):
        """Convert zipped Icentia dataset into individial patient HDF5 files.

        Args:
            zip_path (PathLike): Zipfile path
            patient_ids (npt.NDArray | None, optional): List of patient IDs to extract. Defaults to all.
            force (bool, optional): Whether to force re-download if destination exists. Defaults to False.
            num_workers (int, optional): # parallel workers. Defaults to os.cpu_count().
        """
        if not patient_ids:
            patient_ids = self.patient_ids
        f = functools.partial(self._convert_dataset_pt_zip_to_hdf5, zip_path=zip_path, force=force)
        with Pool(processes=num_workers) as pool:
            _ = list(tqdm(pool.imap(f, patient_ids), total=len(patient_ids)))

    def split_train_test_patients(self, patient_ids: npt.NDArray, test_size: float) -> list[list[int]]:
        """Perform train/test split on patients for given task.
        NOTE: We only perform inter-patient splits and not intra-patient.

        Args:
            patient_ids (npt.NDArray): Patient Ids
            test_size (float): Test size

        Returns:
            list[list[int]]: Train and test sets of patient ids
        """
        # Use stratified split for rhythm task
        if self.task == "rhythm":
            arr_pt_ids = np.intersect1d(self.arr_rhythm_patients, patient_ids)
            norm_pt_ids = np.setdiff1d(patient_ids, arr_pt_ids)
            (
                norm_train_pt_ids,
                norm_val_pt_ids,
            ) = sklearn.model_selection.train_test_split(norm_pt_ids, test_size=test_size)
            (
                arr_train_pt_ids,
                afib_val_pt_ids,
            ) = sklearn.model_selection.train_test_split(arr_pt_ids, test_size=test_size)
            train_pt_ids = np.concatenate((norm_train_pt_ids, arr_train_pt_ids))
            val_pt_ids = np.concatenate((norm_val_pt_ids, afib_val_pt_ids))
            np.random.shuffle(train_pt_ids)
            np.random.shuffle(val_pt_ids)
            return train_pt_ids, val_pt_ids
        # END IF

        # Otherwise, use random split
        return sklearn.model_selection.train_test_split(patient_ids, test_size=test_size)

    @functools.cached_property
    def arr_rhythm_patients(self) -> npt.NDArray:
        """Find all patients with rhythm events. This takes roughly 10 secs.

        Returns:
            npt.NDArray: Patient ids

        """
        patient_ids = self.patient_ids.tolist()
        with Pool() as pool:
            arr_pts_bool = list(pool.imap(self._pt_has_rhythm_arrhythmia, patient_ids))
        patient_ids = np.where(arr_pts_bool)[0]
        return patient_ids

    def _pt_has_rhythm_arrhythmia(self, patient_id: int):
        pt_key = self._pt_key(patient_id)
        with h5py.File(self.ds_path / f"{pt_key}.h5", mode="r") as h5:
            for _, segment in h5[pt_key].items():
                rlabels = segment["rlabels"][:]
                if not rlabels.shape[0]:
                    continue
                rlabels = rlabels[:, 1]
                if len(np.where((rlabels == IcentiaRhythm.afib) | (rlabels == IcentiaRhythm.aflut))[0]):
                    return True
            return False
