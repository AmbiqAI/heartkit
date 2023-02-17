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
from typing import Dict, List, Optional, Tuple, Union

import boto3
import h5py
import numpy as np
import numpy.typing as npt
import pandas as pd
import sklearn.model_selection
import sklearn.preprocessing
from botocore import UNSIGNED
from botocore.client import Config
from tqdm import tqdm

from ..defines import HeartBeat, HeartRate, HeartRhythm, HeartTask
from ..utils import download_file
from .dataset import EcgDataset
from .defines import PatientGenerator, SampleGenerator

logger = logging.getLogger(__name__)


class IcentiaRhythm(IntEnum):
    """Icentia Rhythm labels"""

    noise = 0
    normal = 1
    afib = 2
    aflut = 3
    end = 4
    unknown = 5

    @classmethod
    def hi_priority(cls) -> List[int]:
        """High priority labels"""
        return [cls.afib, cls.aflut]

    @classmethod
    def lo_priority(cls) -> List[int]:
        """Low priority labels"""
        return [cls.noise, cls.normal, cls.end, cls.unknown]


class IcentiaBeat(IntEnum):
    """Incentia beat labels"""

    undefined = 0
    normal = 1
    pac = 2
    # aberrated = 3
    pvc = 4

    @classmethod
    def hi_priority(cls) -> List[int]:
        """High priority labels"""
        return [cls.pac, cls.pvc]

    @classmethod
    def lo_priority(cls) -> List[int]:
        """Low priority labels"""
        return [cls.undefined, cls.normal]


class IcentiaHeartRate(IntEnum):
    """Icentia heart rate labels"""

    tachycardia = 0
    bradycardia = 1
    normal = 2
    noise = 3


##
# These map Icentia specific labels to common labels
##
HeartRhythmMap = {
    IcentiaRhythm.noise: HeartRhythm.noise,
    IcentiaRhythm.normal: HeartRhythm.normal,
    IcentiaRhythm.afib: HeartRhythm.afib,
    IcentiaRhythm.aflut: HeartRhythm.afib,
}

HeartBeatMap = {
    IcentiaBeat.undefined: HeartBeat.noise,
    IcentiaBeat.normal: HeartBeat.normal,
    IcentiaBeat.pac: HeartBeat.pac,
    # IcentiaBeat.aberrated: HeartBeat.pac,
    IcentiaBeat.pvc: HeartBeat.pvc,
}


class IcentiaDataset(EcgDataset):
    """Icentia dataset"""

    def __init__(
        self, ds_path: str, task: HeartTask = HeartTask.rhythm, frame_size: int = 1250
    ) -> None:
        super().__init__(os.path.join(ds_path, "icentia11k"), task, frame_size)

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
    def patient_ids(self) -> npt.ArrayLike:
        """Get dataset patient IDs

        Returns:
            npt.ArrayLike: patient IDs
        """
        return np.arange(11_000)

    def get_train_patient_ids(self) -> npt.ArrayLike:
        """Get training patient IDs

        Returns:
            npt.ArrayLike: patient IDs
        """
        return self.patient_ids[:10_000]

    def get_test_patient_ids(self) -> npt.ArrayLike:
        """Get patient IDs reserved for testing only

        Returns:
            npt.ArrayLike: patient IDs
        """
        return self.patient_ids[10_000:]

    @functools.cached_property
    def arr_rhythm_patients(self) -> npt.ArrayLike:
        """Find all patients with AFIB/AFLUT events. This takes roughly 10 secs.
        Returns:
            npt.ArrayLike: Patient ids
        """
        patient_ids = self.patient_ids.tolist()
        with Pool() as pool:
            arr_pts_bool = list(pool.imap(self._pt_has_rhythm_arrhythmia, patient_ids))
        patient_ids = np.where(arr_pts_bool)[0]
        return patient_ids

    def task_data_generator(
        self,
        patient_generator: PatientGenerator,
        samples_per_patient: Union[int, List[int]] = 1,
    ) -> SampleGenerator:
        """Task-level data generator.

        Args:
            patient_generator (PatientGenerator): Patient data generator
            samples_per_patient (Union[int, List[int]], optional): # samples per patient. Defaults to 1.

        Returns:
            SampleGenerator: Sample data generator
        """
        if self.task == HeartTask.rhythm:
            return self.rhythm_data_generator(
                patient_generator=patient_generator,
                samples_per_patient=samples_per_patient,
            )
        if self.task == HeartTask.beat:
            return self.beat_data_generator(
                patient_generator=patient_generator,
                samples_per_patient=samples_per_patient,
            )
        if self.task == HeartTask.hr:
            return self.heart_rate_data_generator(
                patient_generator=patient_generator,
                samples_per_patient=samples_per_patient,
            )
        raise NotImplementedError()

    def _split_train_test_patients(
        self, patient_ids: npt.ArrayLike, test_size: float
    ) -> List[List[int]]:
        """Split dataset into training and validation. We customize based on task.
            NOTE: We only perform inter-patient splits and not intra-patient.
        Args:
            patient_ids (npt.ArrayLike): Patient ids
            test_size (float): # or proportion of patients to
            task (HeartTask, optional): Task. Defaults to HeartTask.rhythm.

        Returns:
            List[npt.ArrayLike, npt.ArrayLike]: Training and validation patient IDs
        """

        if self.task == HeartTask.rhythm:
            arr_pt_ids = np.intersect1d(self.arr_rhythm_patients, patient_ids)
            norm_pt_ids = np.setdiff1d(patient_ids, arr_pt_ids)
            (
                norm_train_pt_ids,
                norm_val_pt_ids,
            ) = sklearn.model_selection.train_test_split(
                norm_pt_ids, test_size=test_size
            )
            (
                arr_train_pt_ids,
                afib_val_pt_ids,
            ) = sklearn.model_selection.train_test_split(
                arr_pt_ids, test_size=test_size
            )
            train_pt_ids = np.concatenate((norm_train_pt_ids, arr_train_pt_ids))
            val_pt_ids = np.concatenate((norm_val_pt_ids, afib_val_pt_ids))
            np.random.shuffle(train_pt_ids)
            np.random.shuffle(val_pt_ids)
            return train_pt_ids, val_pt_ids
        # END IF
        return sklearn.model_selection.train_test_split(
            patient_ids, test_size=test_size
        )

    def rhythm_data_generator(
        self,
        patient_generator: PatientGenerator,
        samples_per_patient: Union[int, List[int]] = 1,
    ) -> SampleGenerator:
        """Generate frames and rhythm label using patient generator.
        Args:
            patient_generator (PatientGenerator): Patient Generator
            frame_size (int, optional): Size of frame. Defaults to 2048.
            samples_per_patient (Union[int, List[int]], optional): # samples per patient. Defaults to 1.

        Returns:
            SampleGenerator: Sample generator

        Yields:
            Iterator[SampleGenerator]
        """

        tgt_rhythm_labels = (
            IcentiaRhythm.normal,
            IcentiaRhythm.afib,
            IcentiaRhythm.aflut,
        )
        if isinstance(samples_per_patient, Iterable):
            samples_per_tgt = samples_per_patient
        else:
            samples_per_tgt = int(
                max(1, samples_per_patient / len(tgt_rhythm_labels))
            ) * [len(tgt_rhythm_labels)]

        # Group patient rhythms by type (segment, start, stop)
        for _, segments in patient_generator:
            seg_label_map: Dict[str, List[Tuple[str, int, int]]] = {
                lbl: [] for lbl in tgt_rhythm_labels
            }
            for seg_key, segment in segments.items():
                rlabels = segment["rlabels"][:]
                if not rlabels.shape[0]:
                    continue  # Segment has no rhythm labels
                rlabels = rlabels[
                    np.where(rlabels[:, 1] != IcentiaRhythm.noise.value)[0]
                ]
                for i, l in enumerate(rlabels[::2, 1]):
                    xs, xe = rlabels[i * 2 + 0, 0], rlabels[i * 2 + 1, 0]
                    xs += random.randint(0, self.sampling_rate)
                    seg_frame_size = xe - xs + 1
                    if l in tgt_rhythm_labels and (seg_frame_size > self.frame_size):
                        seg_label_map[l].append((seg_key, xs, xe))
                    # END IF
                # END FOR
            # END FOR

            # Grab target segments
            seg_samples: List[Tuple[str, int, int, int]] = []
            for i, label in enumerate(tgt_rhythm_labels):
                tgt_segments = seg_label_map.get(label, [])
                if not tgt_segments:
                    continue
                tgt_seg_indices: List[int] = random.choices(
                    list(range(len(tgt_segments))),
                    weights=[s[2] - s[1] for s in tgt_segments],
                    k=samples_per_tgt[i],
                )
                for tgt_seg_index in tgt_seg_indices:
                    seg_key, rhy_start, rhy_end = tgt_segments[tgt_seg_index]
                    frame_start = np.random.randint(
                        rhy_start, rhy_end - self.frame_size + 1
                    )
                    frame_end = frame_start + self.frame_size
                    seg_samples.append(
                        (seg_key, frame_start, frame_end, HeartRhythmMap[label])
                    )
                # END FOR
            # END FOR

            # Shuffle segments
            random.shuffle(seg_samples)
            for seg_key, frame_start, frame_end, label in seg_samples:
                x: npt.NDArray = segments[seg_key]["data"][
                    frame_start:frame_end
                ].astype(np.float32)
                yield x, label
            # END FOR
        # END FOR

    def beat_data_generator(
        self,
        patient_generator: PatientGenerator,
        samples_per_patient: int = 1,
    ) -> SampleGenerator:
        """Generate frames and beat label using patient generator.
        There are over 2.5 billion normal and undefined while less than 40 million arrhythmia beats.
        The following routine sorts each patient's beats by type and then approx. uniformly samples them by amount requested.
        We start with arrhythmia types followed by undefined and normal. For each beat we resplit remaining samples requested.
        Args:
            patient_generator (PatientGenerator): Patient generator
            frame_size (int, optional): Frame size. Defaults to 2048.
            samples_per_patient (int, optional): # samples per patient. Defaults to 1.

        Returns:
            SampleGenerator: Sample generator

        Yields:
            Iterator[SampleGenerator]
        """
        tgt_beat_labels = [
            IcentiaBeat.pvc,
            IcentiaBeat.pac,
            IcentiaBeat.undefined,
            IcentiaBeat.normal,
        ]
        for _, segments in patient_generator:
            # This maps segment index to segment key
            seg_map: List[str] = list(segments.keys())

            num_rem_samples = samples_per_patient
            num_rem_beats = len(tgt_beat_labels)

            # For each beat type, locate all beats in segments
            pt_segs_beat_idxs: List[Tuple[int, int, int]] = []
            for beat in tgt_beat_labels:
                beat_segs_idxs: List[Tuple[int, int, int]] = []
                for seg_idx, seg_key in enumerate(seg_map):
                    blabels = segments[seg_key]["blabels"][:]
                    if blabels.shape[0] == 0:
                        continue
                    # NOTE: Could remove beats too close to start or end
                    beat_idxs = np.where(blabels[:, 1] == beat.value)[0].tolist()
                    if len(beat_idxs):
                        beat_segs_idxs += [
                            (seg_idx, beat_idx, beat) for beat_idx in beat_idxs
                        ]
                    # END IF
                # END FOR

                # Shuffle all beats for given beat type
                random.shuffle(beat_segs_idxs)

                # Grab N samples of given beat type
                num_beat_samples = min(
                    int(num_rem_samples / num_rem_beats), len(beat_segs_idxs)
                )
                num_rem_samples -= num_beat_samples
                num_rem_beats -= 1
                if num_beat_samples:
                    beat_segs_idxs = beat_segs_idxs[:num_beat_samples]
                    pt_segs_beat_idxs += beat_segs_idxs
            # END FOR

            random.shuffle(pt_segs_beat_idxs)

            # Yield selected samples for patient
            for seg_idx, beat_idx, beat_label in pt_segs_beat_idxs:
                frame_start = max(0, beat_idx - int(self.frame_size / 2))
                frame_end = frame_start + self.frame_size
                x = segments[seg_map[seg_idx]]["data"][frame_start:frame_end].astype(
                    np.float32
                )
                y = beat_label
                if x.shape[0] != self.frame_size:
                    continue
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
                frame_center = (
                    np.random.randint(segment_size - max_frame_size)
                    + max_frame_size // 2
                )
                signal_frame_start = frame_center - self.frame_size // 2
                signal_frame_end = frame_center + self.frame_size // 2
                x = segment["data"][signal_frame_start:signal_frame_end]
                label_frame_start = frame_center - label_frame_size // 2
                label_frame_end = frame_center + label_frame_size // 2
                beat_indices = segment["blabels"][:, 0]
                frame_beat_indices = self.get_complete_beats(
                    beat_indices, start=label_frame_start, end=label_frame_end
                )
                y = self._get_heart_rate_label(frame_beat_indices, self.sampling_rate)
                yield x, y
            # END FOR
        # END FOR

    def signal_generator(
        self, patient_generator: PatientGenerator, samples_per_patient: int = 1
    ):
        """
        Generate frames using patient generator.
        from the segments in patient data by placing a frame in a random location within one of the segments.
        Args:
        patient_generator (PatientGenerator): Generator that yields a tuple of patient id and patient data.
                Patient data may contain only signals, since labels are not used.
        samples_per_patient (int): Samples per patient.
        Return: Generator of: input data of shape (frame_size, 1)
        """
        for _, segments in patient_generator:
            for _ in range(samples_per_patient):
                segment = segments[np.random.choice(list(segments.keys()))]
                segment_size = segment["data"].shape[0]
                frame_start = np.random.randint(segment_size - self.frame_size)
                frame_end = frame_start + self.frame_size
                x = segment["data"][frame_start:frame_end]
                yield x
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
                logger.debug("X")
                np.random.shuffle(patient_ids)
            for patient_id in patient_ids:
                pt_key = self._pt_key(patient_id)
                with h5py.File(
                    os.path.join(self.ds_path, f"{pt_key}.h5"), mode="r"
                ) as h5:
                    patient_data = h5[pt_key]
                    yield patient_id, patient_data
            # END FOR
            if not repeat:
                break
        # END WHILE

    def random_patient_generator(
        self,
        patient_ids: List[int],
        patient_weights: Optional[List[int]] = None,
    ) -> PatientGenerator:
        """Samples patient data from the provided patient distribution.

        Args:
            patient_ids (List[int]): Patient ids
            patient_weights (Optional[List[int]], optional): Probabilities associated with each patient. Defaults to None.

        Returns:
            PatientGenerator: Patient generator

        Yields:
            Iterator[PatientGenerator]
        """
        while True:
            for patient_id in np.random.choice(
                patient_ids, size=1024, p=patient_weights
            ):
                pt_key = self._pt_key(patient_id)
                with h5py.File(
                    os.path.join(self.ds_path, f"{pt_key}.h5"), mode="r"
                ) as h5:
                    patient_data = h5[pt_key]
                    yield patient_id, patient_data
            # END FOR
        # END WHILE

    def _pt_key(self, patient_id: int):
        return f"p{patient_id:05d}"

    def get_complete_beats(
        self,
        indices: npt.ArrayLike,
        labels: npt.ArrayLike = None,
        start: int = 0,
        end: Optional[int] = None,
    ):
        """
        Find all complete beats within a frame i.e. start and end of the beat lie within the frame.
        The indices are assumed to specify the end of a heartbeat.
        Args:
            indices (np.ArrayLike): List of sorted beat indices.
            labels (np.ArrayLike): List of beat labels.
            start (int): Index of the first sample in the frame.
            end (Optional[int]): Index of the last sample in the frame.
        Returns:
            Tuple[npt.ArrayLike, npt.ArrayLine] (beat indices, beat labels)
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
        return indices_slice, label_slice

    def _get_rhythm_label(self, durations: npt.ArrayLike, labels: npt.ArrayLike):
        """Determine rhythm label based on the longest rhythm among arrhythmias.
        Args:
            durations (npt.ArrayLike): Array of rhythm durations
            labels (npt.ArrayLike): Array of rhythm labels
        Returns:
            Rhythm label as an integer
        """
        # sum up the durations of each rhythm
        summed_durations = np.zeros(len(IcentiaRhythm))
        for rhythm in IcentiaRhythm:
            summed_durations[rhythm.value] = durations[labels == rhythm.value].sum()
        longest_hp_rhythm = np.argmax(summed_durations[IcentiaRhythm.hi_priority()])
        if summed_durations[IcentiaRhythm.hi_priority()][longest_hp_rhythm] > 0:
            y = HeartRhythmMap[IcentiaRhythm.hi_priority()[longest_hp_rhythm]]
        else:
            longest_lp_rhythm = np.argmax(summed_durations[IcentiaRhythm.lo_priority()])
            # handle the case of no detected rhythm
            if summed_durations[IcentiaRhythm.lo_priority()][longest_lp_rhythm] > 0:
                y = HeartRhythmMap[IcentiaRhythm.lo_priority()[longest_lp_rhythm]]
            else:
                y = HeartRhythmMap[IcentiaRhythm.noise]
        return y

    def _get_beat_label(self, labels: npt.ArrayLike):
        """Determine beat label based on the occurrence of pac / abberated / pvc,
            otherwise pick the most common beat type among the normal / undefined.

        Args:
            labels (List[int]): Array of beat labels.

        Returns:
            int:  Beat label as an integer.
        """
        # calculate the count of each beat type in the frame
        beat_counts = np.bincount(labels, minlength=len(IcentiaBeat))
        max_hp_idx = np.argmax(beat_counts[IcentiaBeat.hi_priority()])
        if beat_counts[IcentiaBeat.hi_priority()][max_hp_idx] > 0:
            y = HeartBeatMap[IcentiaBeat.hi_priority()[max_hp_idx]]
        else:
            max_lp_idx = np.argmax(beat_counts[IcentiaBeat.lo_priority()])
            # handle the case of no detected beats
            if beat_counts[IcentiaBeat.lo_priority()][max_lp_idx] > 0:
                y = HeartBeatMap[IcentiaBeat.lo_priority()[max_lp_idx]]
            else:
                y = HeartBeatMap[IcentiaBeat.undefined]
        return y

    def _get_heart_rate_label(self, qrs_indices, fs=None) -> int:
        """Determine the heart rate label based on an array of QRS indices (separating individual heartbeats).
            The QRS indices are assumed to be measured in seconds if sampling frequency `fs` is not specified.
            The heartbeat label is based on the following BPM (beats per minute) values: (0) tachycardia <60 BPM,
            (1) bradycardia >100 BPM, (2) healthy 60-100 BPM, (3) noisy if QRS detection failed.

        Args:
            qrs_indices (List[int]): Array of QRS indices.
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
            return HeartRate.normal.value
        return HeartRate.tachycardia.value

    def _pt_has_rhythm_arrhythmia(self, patient_id: int):
        pt_key = self._pt_key(patient_id)
        with h5py.File(os.path.join(self.ds_path, f"{pt_key}.h5"), mode="r") as h5:
            for _, segment in h5[pt_key].items():
                rlabels = segment["rlabels"][:]
                if not rlabels.shape[0]:
                    continue
                rlabels = rlabels[:, 1]
                if len(
                    np.where(
                        (rlabels == IcentiaRhythm.afib)
                        | (rlabels == IcentiaRhythm.aflut)
                    )[0]
                ):
                    return True
            return False

    def get_rhythm_statistics(
        self,
        patient_ids: Optional[npt.ArrayLike] = None,
        save_path: Optional[str] = None,
    ) -> pd.DataFrame:
        """Utility function to extract rhythm statistics across entire dataset. Useful for EDA.

        Args:
            patient_ids (Optional[npt.ArrayLike], optional): Patients IDs to include. Defaults to all.
            save_path (Optional[str], optional): Parquet file path to save results. Defaults to None.

        Returns:
            pd.DataFrame: DataFrame of statistics
        """

        if patient_ids is None:
            patient_ids = self.patient_ids
        pt_gen = self.uniform_patient_generator(patient_ids=patient_ids, repeat=False)
        stats = []
        for pt, segments in pt_gen:
            # Group patient rhythms by type (segment, start, stop)
            segment_label_map: Dict[str, List[Tuple[str, int, int]]] = {}
            for seg_key, segment in segments.items():
                rlabels = segment["rlabels"][:]
                if rlabels.shape[0] == 0:
                    continue  # Segment has no rhythm labels
                rlabels = rlabels[
                    np.where(rlabels[:, 1] != IcentiaRhythm.noise.value)[0]
                ]
                for i, l in enumerate(rlabels[::2, 1]):
                    if l in (
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

    def download(self, num_workers: Optional[int] = None, force: bool = False):
        """Download dataset

        Args:
            num_workers (Optional[int], optional): # parallel workers. Defaults to None.
            force (bool, optional): Force redownload. Defaults to False.
        """

        def download_s3_file(
            s3_file: str,
            save_path: str,
            bucket: str,
            client: boto3.client,
            force: bool = False,
        ):
            if not force and os.path.exists(save_path):
                return
            client.download_file(
                Bucket=bucket,
                Key=s3_file,
                Filename=save_path,
            )

        s3_bucket = "ambiqai-ecg-icentia11k-dataset"
        s3_prefix = "patients"

        os.makedirs(self.ds_path, exist_ok=True)

        patient_ids = self.patient_ids

        # Creating only one session and one client
        session = boto3.Session()
        client = session.client("s3", config=Config(signature_version=UNSIGNED))

        func = functools.partial(
            download_s3_file, bucket=s3_bucket, client=client, force=force
        )

        with tqdm(
            desc="Downloading icentia11k dataset from S3", total=len(patient_ids)
        ) as pbar:
            pt_keys = [self._pt_key(patient_id) for patient_id in patient_ids]
            with ThreadPoolExecutor(max_workers=2 * num_workers) as executor:
                futures = (
                    executor.submit(
                        func,
                        f"{s3_prefix}/{pt_key}.h5",
                        os.path.join(self.ds_path, f"{pt_key}.h5"),
                    )
                    for pt_key in pt_keys
                )
                for future in as_completed(futures):
                    err = future.exception()
                    if err:
                        print("Failed on file", err)
                    pbar.update(1)
                # END FOR
            # END WITH
        # END WITH

    def download_raw_dataset(
        self, num_workers: Optional[int] = None, force: bool = False
    ):
        """Downloads full Icentia dataset zipfile and converts into individial patient HDF5 files.
        NOTE: This is a very long process (e.g. 24 hrs). Please use `icentia11k.download_dataset` instead.
        Args:
            zip_path (str): Zipfile path
            patient_ids (Optional[npt.ArrayLike], optional): List of patient IDs to extract. Defaults to all.
            force (bool, optional): Whether to force re-download if destination exists. Defaults to False.
            num_workers (int, optional): # parallel workers. Defaults to os.cpu_count().
        """
        logger.info("Downloading icentia11k dataset")
        ds_url = (
            "https://physionet.org/static/published-projects/icentia11k-continuous-ecg/"
            "icentia11k-single-lead-continuous-raw-electrocardiogram-dataset-1.0.zip"
        )
        ds_zip_path = os.path.join(self.ds_path, "icentia11k.zip")
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

    def _convert_dataset_pt_zip_to_hdf5(
        self, patient: int, zip_path: str, force: bool = False
    ):
        """Extract patient data from Icentia zipfile. Pulls out ECG data along with all labels.

        Args:
            patient (int): Patient id
            zip_path (str): Zipfile path
            force (bool, optional): Whether to override destination if it exists. Defaults to False.
        """
        import re  # pylint: disable=import-outside-toplevel

        import wfdb  # pylint: disable=import-outside-toplevel

        # These map Wfdb labels to icentia labels
        WfdbRhythmMap = {"": 0, "(N": 1, "(AFIB": 2, "(AFL": 3, ")": 4}
        WfdbBeatMap = {"Q": 0, "N": 1, "S": 2, "a": 3, "V": 4}

        logger.info(f"Processing patient {patient}")
        pt_id = self._pt_key(patient)
        pt_path = os.path.join(self.ds_path, f"{pt_id}.h5")
        if not force and os.path.exists(pt_path):
            print("skipping patient")
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
                    [
                        [atr.sample[i], WfdbBeatMap.get(s)]
                        for i, s in enumerate(atr.symbol)
                        if s in WfdbBeatMap
                    ],
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
                print(f"Failed processing {zp_rec_name}", err)
                continue
        h5.close()

    def _convert_dataset_zip_to_hdf5(
        self,
        zip_path: str,
        patient_ids: Optional[npt.ArrayLike] = None,
        force: bool = False,
        num_workers: Optional[int] = None,
    ):
        """Convert zipped Icentia dataset into individial patient HDF5 files.

        Args:
            zip_path (str): Zipfile path
            patient_ids (Optional[npt.ArrayLike], optional): List of patient IDs to extract. Defaults to all.
            force (bool, optional): Whether to force re-download if destination exists. Defaults to False.
            num_workers (int, optional): # parallel workers. Defaults to os.cpu_count().
        """
        if not patient_ids:
            patient_ids = self.patient_ids
        f = functools.partial(
            self._convert_dataset_pt_zip_to_hdf5, zip_path=zip_path, force=force
        )
        with Pool(processes=num_workers) as pool:
            _ = list(tqdm(pool.imap(f, patient_ids), total=len(patient_ids)))
