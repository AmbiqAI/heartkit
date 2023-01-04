import functools
import logging
import os
import random
import tempfile
import warnings
import zipfile
from collections.abc import Iterable
from enum import IntEnum
from multiprocessing import Pool
from typing import Dict, List, Optional, Tuple, Union

import h5py
import numpy as np
import numpy.typing as npt
import pandas as pd
import sklearn.model_selection
import sklearn.preprocessing
from tqdm import tqdm

from ..types import EcgTask, HeartBeat, HeartRate, HeartRhythm
from .types import PatientGenerator, SampleGenerator
from .utils import butter_bp_filter

logger = logging.getLogger("ECGARR")

# Patients containing AFIB/AFLUT events
# fmt: off
arr_rhythm_patients = [
    16,    20,    53,    60,    65,    75,    84,    91,   119,
    139,   148,   159,   166,   177,   198,   230,   247,   268,
    271,   281,   287,   292,   295,   299,   303,   323,   328,
    337,   365,   404,   417,   418,   434,   446,   456,   457,
    462,   464,   471,   484,   487,   499,   507,   508,   529,
    534,   535,   580,   584,   591,   614,   618,   636,   680,
    719,   809,   825,   831,   832,   836,   843,   886,   903,
    907,   911,   922,   957,   963,   967,   1022,  1034,  1041,
    1066,  1100,  1117,  1124,  1162,  1163,  1166,  1215,  1219,
    1221,  1231,  1233,  1251,  1271,  1281,  1293,  1325,  1331,
    1340,  1342,  1361,  1386,  1397,  1416,  1420,  1443,  1461,
    1503,  1528,  1530,  1543,  1556,  1562,  1609,  1620,  1624,
    1628,  1634,  1645,  1673,  1679,  1680,  1693,  1698,  1705,
    1738,  1744,  1749,  1753,  1781,  1807,  1809,  1824,  1836,
    1847,  1848,  1850,  1892,  1894,  1904,  1933,  1934,  1949,
    1964,  1975,  1978,  1980,  1989,  2001,  2015,  2017,  2050,
    2093,  2108,  2125,  2134,  2140,  2155,  2166,  2188,  2224,
    2231,  2240,  2252,  2255,  2269,  2298,  2361,  2362,  2404,
    2428,  2478,  2479,  2496,  2499,  2508,  2521,  2541,  2569,
    2590,  2601,  2643,  2648,  2650,  2653,  2664,  2679,  2680,
    2690,  2701,  2710,  2712,  2748,  2753,  2760,  2767,  2773,
    2837,  2842,  2844,  2845,  2860,  2862,  2867,  2869,  2871,
    2878,  2884,  2903,  2906,  2908,  2917,  2959,  2968,  2991,
    3017,  3024,  3039,  3047,  3058,  3059,  3066,  3076,  3085,
    3086,  3126,  3169,  3179,  3183,  3210,  3217,  3218,  3232,
    3235,  3264,  3266,  3283,  3287,  3288,  3293,  3324,  3330,
    3348,  3369,  3370,  3386,  3400,  3404,  3410,  3424,  3426,
    3456,  3458,  3468,  3484,  3485,  3490,  3499,  3523,  3528,
    3531,  3560,  3594,  3638,  3648,  3650,  3662,  3666,  3677,
    3693,  3698,  3706,  3720,  3725,  3728,  3730,  3745,  3749,
    3751,  3765,  3834,  3871,  3881,  3882,  3905,  3909,  3910,
    3927,  3946,  3949,  3956,  3982,  3985,  3991,  4007,  4025,
    4030,  4035,  4041,  4050,  4068,  4086,  4096,  4103,  4122,
    4128,  4152,  4226,  4240,  4248,  4263,  4267,  4282,  4283,
    4294,  4301,  4308,  4314,  4324,  4331,  4333,  4341,  4353,
    4354,  4355,  4396,  4397,  4401,  4411,  4417,  4424,  4429,
    4455,  4459,  4488,  4497,  4506,  4516,  4538,  4561,  4567,
    4568,  4572,  4574,  4617,  4618,  4620,  4621,  4629,  4647,
    4652,  4661,  4685,  4687,  4716,  4721,  4753,  4759,  4773,
    4776,  4793,  4815,  4834,  4838,  4862,  4884,  4892,  4915,
    4936,  4983,  4986,  5007,  5065,  5081,  5087,  5094,  5103,
    5111,  5125,  5160,  5162,  5163,  5184,  5223,  5234,  5296,
    5297,  5300,  5306,  5348,  5354,  5355,  5361,  5380,  5407,
    5447,  5453,  5469,  5476,  5488,  5555,  5559,  5595,  5599,
    5604,  5621,  5629,  5670,  5672,  5708,  5715,  5716,  5719,
    5739,  5741,  5824,  5827,  5839,  5845,  5856,  5865,  5867,
    5895,  5901,  5902,  5922,  5934,  5935,  5963,  5968,  5982,
    6002,  6026,  6041,  6043,  6072,  6074,  6081,  6083,  6084,
    6120,  6122,  6129,  6172,  6212,  6218,  6248,  6249,  6266,
    6270,  6292,  6294,  6298,  6355,  6360,  6373,  6375,  6381,
    6389,  6408,  6441,  6446,  6447,  6492,  6493,  6503,  6504,
    6521,  6523,  6535,  6542,  6575,  6594,  6597,  6606,  6665,
    6671,  6681,  6697,  6716,  6722,  6753,  6754,  6755,  6756,
    6763,  6776,  6803,  6845,  6895,  6900,  6923,  6947,  6949,
    6969,  6978,  6994,  7024,  7040,  7043,  7072,  7073,  7075,
    7095,  7116,  7139,  7152,  7153,  7175,  7186,  7188,  7189,
    7192,  7198,  7211,  7232,  7236,  7249,  7271,  7277,  7308,
    7328,  7359,  7368,  7378,  7380,  7390,  7391,  7434,  7459,
    7462,  7489,  7503,  7508,  7512,  7553,  7570,  7571,  7589,
    7612,  7638,  7653,  7668,  7684,  7686,  7710,  7713,  7715,
    7721,  7730,  7749,  7786,  7790,  7804,  7809,  7822,  7825,
    7839,  7846,  7863,  7893,  7897,  7905,  7950,  7964,  7968,
    7984,  8008,  8009,  8025,  8092,  8098,  8101,  8106,  8114,
    8141,  8144,  8162,  8193,  8195,  8212,  8222,  8233,  8241,
    8282,  8289,  8295,  8329,  8335,  8353,  8357,  8392,  8398,
    8412,  8455,  8473,  8500,  8514,  8532,  8547,  8559,  8582,
    8599,  8600,  8640,  8651,  8689,  8718,  8736,  8773,  8820,
    8836,  8838,  8840,  8851,  8853,  8866,  8900,  8975,  9026,
    9123,  9157,  9158,  9160,  9164,  9173,  9183,  9210,  9216,
    9234,  9254,  9257,  9282,  9284,  9302,  9309,  9318,  9322,
    9331,  9351,  9366,  9383,  9400,  9420,  9468,  9475,  9476,
    9484,  9493,  9495,  9536,  9574,  9600,  9635,  9704,  9705,
    9741,  9747,  9764,  9779,  9784,  9788,  9795,  9803,  9839,
    9849,  9855,  9867,  9868,  9909,  9915,  9942,  9965,  9968,
    9976,  9977,  10034, 10048, 10070, 10103, 10112, 10151, 10160,
    10172, 10184, 10188, 10195, 10198, 10201, 10206, 10211, 10212,
    10224, 10228, 10248, 10259, 10268, 10274, 10284, 10293, 10303,
    10339, 10340, 10344, 10375, 10390, 10396, 10397, 10402, 10430,
    10449, 10450, 10462, 10476, 10491, 10519, 10528, 10541, 10544,
    10573, 10576, 10600, 10602, 10605, 10615, 10619, 10620, 10629,
    10672, 10687, 10694, 10702, 10726, 10750, 10759, 10760, 10764,
    10778, 10784, 10812, 10813, 10839, 10852, 10853, 10915, 10949,
    10951, 10958, 10961, 10966, 10969, 10974, 10979, 10994, 10995
]
# fmt: on


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
    aberrated = 3
    pvc = 4

    @classmethod
    def hi_priority(cls) -> List[int]:
        """High priority labels"""
        return [cls.pac, cls.aberrated, cls.pvc]

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
# These map Wfdb labels to icentia labels
##

WfdbRhythmMap = {"": 0, "(N": 1, "(AFIB": 2, "(AFL": 3, ")": 4}

WfdbBeatMap = {"Q": 0, "N": 1, "S": 2, "a": 3, "V": 4}

##
# These map Icentia specific labels to common arrhythmia labels
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
    IcentiaBeat.aberrated: HeartBeat.pac,
    IcentiaBeat.pvc: HeartBeat.pvc,
}

ds_sampling_rate = 250  # Hz
ds_mean = 0.0018
ds_std = 1.3711


def get_patient_ids() -> npt.ArrayLike:
    """Get dataset patient IDs

    Returns:
        npt.ArrayLike: patient IDs
    """
    return np.arange(11_000)


def get_train_patient_ids() -> npt.ArrayLike:
    """Get dataset training patient IDs

    Returns:
        npt.ArrayLike: patient IDs
    """
    return get_patient_ids()[:10_000]


def get_test_patient_ids() -> npt.ArrayLike:
    """Get dataset patient IDs reserved for testing only

    Returns:
        npt.ArrayLike: patient IDs
    """
    return get_patient_ids()[10_000:]


def train_test_split_patients(
    patient_ids: npt.ArrayLike, test_size: float, task: EcgTask = EcgTask.rhythm
) -> Tuple[npt.ArrayLike, npt.ArrayLike]:
    """Split dataset into training and validation. We customize based on task.
        NOTE: We only perform inter-patient splits and not intra-patient.
    Args:
        patient_ids (npt.ArrayLike): Patient ids
        test_size (float): # or proportion of patients to
        task (EcgTask, optional): _description_. Defaults to EcgTask.rhythm.

    Returns:
        List[npt.ArrayLike, npt.ArrayLike]: Training and validation patient IDs
    """
    if task == EcgTask.rhythm:
        arr_pt_ids = np.intersect1d(np.array(arr_rhythm_patients), patient_ids)
        norm_pt_ids = np.setdiff1d(patient_ids, arr_pt_ids)
        norm_train_pt_ids, norm_val_pt_ids = sklearn.model_selection.train_test_split(
            norm_pt_ids, test_size=test_size
        )
        arr_train_pt_ids, afib_val_pt_ids = sklearn.model_selection.train_test_split(
            arr_pt_ids, test_size=test_size
        )
        train_pt_ids = np.concatenate((norm_train_pt_ids, arr_train_pt_ids))
        val_pt_ids = np.concatenate((norm_val_pt_ids, afib_val_pt_ids))
        np.random.shuffle(train_pt_ids)
        np.random.shuffle(val_pt_ids)
        return train_pt_ids, val_pt_ids
    # END IF
    return sklearn.model_selection.train_test_split(patient_ids, test_size=test_size)


def rhythm_data_generator(
    patient_generator: PatientGenerator,
    frame_size: int = 2048,
    samples_per_patient: Union[int, List[int]] = 1,
) -> SampleGenerator:
    """Generate a stream of short signals and their corresponding rhythm label. These short signals are uniformly sampled
        from the segments in patient data by placing a frame in a random location within one of the segments.
        The corresponding label is then determined based on the rhythm durations within this frame.

    Args:
        patient_generator (PatientGenerator): Patient Generator
        frame_size (int, optional): Size of frame. Defaults to 2048.
        samples_per_patient (Union[int, List[int]], optional): # samples per patient. Defaults to 1.

    Returns:
        SampleGenerator: Sample generator

    Yields:
        Iterator[SampleGenerator]
    """

    tgt_rhythm_labels = (IcentiaRhythm.normal, IcentiaRhythm.afib, IcentiaRhythm.aflut)
    if isinstance(samples_per_patient, Iterable):
        samples_per_tgt = samples_per_patient
    else:
        samples_per_tgt = int(max(1, samples_per_patient / len(tgt_rhythm_labels))) * [
            len(tgt_rhythm_labels)
        ]
    for _, segments in patient_generator:
        # Group patient rhythms by type (segment, start, stop)
        seg_label_map: Dict[str, List[Tuple[str, int, int]]] = {
            lbl: [] for lbl in tgt_rhythm_labels
        }
        for seg_key, segment in segments.items():
            rlabels = segment["rlabels"][:]
            if not rlabels.shape[0]:
                continue  # Segment has no rhythm labels
            rlabels = rlabels[np.where(rlabels[:, 1] != IcentiaRhythm.noise.value)[0]]
            for i, l in enumerate(rlabels[::2, 1]):
                xs, xe = rlabels[i * 2 + 0, 0], rlabels[i * 2 + 1, 0]
                xs += random.randint(0, ds_sampling_rate)
                seg_frame_size = xe - xs + 1
                if l in tgt_rhythm_labels and (seg_frame_size > frame_size):
                    seg_label_map[l].append((seg_key, xs, xe))
                # END IF
            # END FOR
        # END FOR

        seg_samples: List[Tuple[str, int, int, int]] = []

        # Grab target segments
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
                frame_start = np.random.randint(rhy_start, rhy_end - frame_size + 1)
                frame_end = frame_start + frame_size
                seg_samples.append(
                    (seg_key, frame_start, frame_end, HeartRhythmMap[label])
                )
            # END FOR
        # END FOR

        # Shuffle segments
        random.shuffle(seg_samples)

        for seg_key, frame_start, frame_end, label in seg_samples:
            x: npt.NDArray = segments[seg_key]["data"][frame_start:frame_end].astype(
                np.float32
            )
            yield x, label
        # END FOR
    # END FOR


def beat_data_generator(
    patient_generator: PatientGenerator,
    frame_size: int = 2048,
    samples_per_patient: int = 1,
) -> SampleGenerator:
    """Generate a stream of short signals and their corresponding beat label. These short signals are uniformly sampled
        from the segments in patient data by placing a frame in a random location within one of the segments.
        The corresponding label is then determined based on the beats within this frame.

    Args:
        patient_generator (PatientGenerator): Patient generator
        frame_size (int, optional): Frame size. Defaults to 2048.
        samples_per_patient (int, optional): # samples per patient. Defaults to 1.

    Returns:
        SampleGenerator: Sample generator

    Yields:
        Iterator[SampleGenerator]
    """
    for _, segments in patient_generator:
        for _ in range(samples_per_patient):
            segment = segments[np.random.choice(list(segments.keys()))]
            segment_size: int = segment["data"].shape[0]
            frame_start = np.random.randint(segment_size - frame_size)
            frame_end = frame_start + frame_size
            if not segment["blabels"].shape[0]:
                continue
            x = segment["data"][frame_start:frame_end]
            beat_indices, beat_labels = (
                segment["blabels"][:, 0],
                segment["blabels"][:, 1],
            )
            # calculate the count of each beat type in the frame and determine the final label
            _, frame_beat_labels = get_complete_beats(
                beat_indices, beat_labels, frame_start, frame_end
            )
            y = get_beat_label(frame_beat_labels)
            yield x, y
        # END FOR
    # END FOR


def heart_rate_data_generator(
    patient_generator: PatientGenerator,
    frame_size: int = 2048,
    label_frame_size: Optional[int] = None,
    samples_per_patient=1,
) -> SampleGenerator:
    """Generate a stream of short signals and their corresponding heart rate label. These short signals are uniformly
        sampled from the segments in patient data by placing a frame in a random location within one of the segments.
        The corresponding label is then determined based on the beats within this frame.

    Args:
        patient_generator (PatientGenerator): Patient generator
        frame_size (int, optional): Frame size. Defaults to 2048.
        label_frame_size (Optional[int], optional): Size of the frame centered on the input signal frame,
            that contains a short signal used for determining the label.
            By default equal to the size of the input signal frame. Defaults to None.
        samples_per_patient (int, optional): # samples per patient. Defaults to 1.

    Returns:
        SampleGenerator: Sample generator

    Yields:
        Iterator[SampleGenerator]
    """
    if label_frame_size is None:
        label_frame_size = frame_size
    max_frame_size = max(frame_size, label_frame_size)
    for _, segments in patient_generator:
        for _ in range(samples_per_patient):
            segment = segments[np.random.choice(list(segments.keys()))]
            segment_size: int = segment["data"].shape[0]
            frame_center = (
                np.random.randint(segment_size - max_frame_size) + max_frame_size // 2
            )
            signal_frame_start = frame_center - frame_size // 2
            signal_frame_end = frame_center + frame_size // 2
            x = segment["data"][signal_frame_start:signal_frame_end]
            label_frame_start = frame_center - label_frame_size // 2
            label_frame_end = frame_center + label_frame_size // 2
            beat_indices = segment["blabels"][:, 0]
            frame_beat_indices = get_complete_beats(
                beat_indices, start=label_frame_start, end=label_frame_end
            )
            y = get_heart_rate_label(frame_beat_indices, ds_sampling_rate)
            yield x, y
        # END FOR
    # END FOR


def signal_generator(patient_generator, frame_size=2048, samples_per_patient=1):
    """
    Generate a stream of short signals. These short signals are uniformly sampled
    from the segments in patient data by placing a frame in a random location within one of the segments.
    @param patient_generator: Generator that yields a tuple of patient id and patient data at each iteration.
            Patient data may contain only signals, since labels are not used.
    @param frame_size: Size of the frame that contains a short signal.
    @param samples_per_patient: Number of samples from one patient before new patient is pulled from the generator.
            This is done in order to decrease the number of i/o operations.
    @return: Generator of: input data of shape (frame_size, 1)
    """
    for _, segments in patient_generator:
        for _ in range(samples_per_patient):
            segment = segments[np.random.choice(list(segments.keys()))]
            segment_size = segment["data"].shape[0]
            frame_start = np.random.randint(segment_size - frame_size)
            frame_end = frame_start + frame_size
            x = segment["data"][frame_start:frame_end]
            yield x
        # END FOR
    # END FOR


def uniform_patient_generator(
    db_path: str,
    patient_ids: npt.ArrayLike,
    repeat: bool = True,
    shuffle: bool = True,
) -> PatientGenerator:
    """Yield data for each patient in the array.

    Args:
        db_path (str): Database path
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
            pt_key = f"p{patient_id:05d}"
            with h5py.File(os.path.join(db_path, f"{pt_key}.h5"), mode="r") as h5:
                patient_data = h5[pt_key]
                yield patient_id, patient_data
        # END FOR
        if not repeat:
            break
    # END WHILE


def random_patient_generator(
    db_path: str,
    patient_ids: List[int],
    patient_weights: Optional[List[int]] = None,
) -> PatientGenerator:
    """Samples patient data from the provided patient distribution.

    Args:
        db_path (str): Database path
        patient_ids (List[int]): Patient ids
        patient_weights (Optional[List[int]], optional): Probabilities associated with each patient. Defaults to None.

    Returns:
        PatientGenerator: Patient generator

    Yields:
        Iterator[PatientGenerator]
    """
    while True:
        for patient_id in np.random.choice(patient_ids, size=1024, p=patient_weights):
            pt_key = f"p{patient_id:05d}"
            with h5py.File(os.path.join(db_path, f"{pt_key}.h5"), mode="r") as h5:
                patient_data = h5[pt_key]
                yield patient_id, patient_data
            # END WITH
        # END FOR
    # END WHILE


def count_labels(labels: List[Tuple[int, int]], num_classes: int) -> npt.ArrayLike:
    """Count the number of labels in all segments.
    Args:
        labels (List[Tuple[int, int]]): Array of tuples of indices, labels.
            Each tuple contains the labels within a segment.
        num_classes (int):  Number of classes (either beat or rhythm depending on the label type).

    Returns:
        npt.ArrayLike: Label counts of shape (num_segments, num_classes)
    """
    return np.array(
        [
            np.bincount(segment_labels, minlength=num_classes)
            for _, segment_labels in labels
        ]
    )


def calculate_durations(
    labels: List[Tuple[int, int]], num_classes: int
) -> npt.ArrayLike:
    """Calculate the duration of each label in all segments.
    Args:
        labels (List[Tuple[int, int]]): Array of tuples of indices, labels.
            Each tuple corresponds to a segment.
        num_classes (int): Number of classes (either beat or rhythm depending on the label type)

    Returns:
        npt.ArrayLike: Label durations of shape (num_segments, num_classes)
    """
    num_segments = len(labels)
    durations = np.zeros((num_segments, num_classes), dtype="int32")
    for segment_index, (segment_indices, segment_labels) in enumerate(labels):
        segment_durations = np.diff(segment_indices, prepend=0)
        for label in range(num_classes):
            durations[segment_index, label] = segment_durations[
                segment_labels == label
            ].sum()
        # END FOR
    # END FOR
    return durations


def get_complete_beats(indices, labels=None, start=0, end=None):
    """
    Find all complete beats within a frame i.e. start and end of the beat lie within the frame.
    The indices are assumed to specify the end of a heartbeat.
    @param indices: Array of beat indices. Indices are assumed to be sorted.
    @param labels: Array of beat labels.
    @param start: Index of the first sample in the frame.
    @param end: Index of the last sample in the frame. By default the last element in the indices array.
    @return: Tuple of: (beat indices, beat labels) in the provided frame
    or only beat indices if labels are not provided.
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


def get_rhythm_label(durations, labels):
    """
    Determine rhythm label based on the longest rhythm among arrhythmias if present,
    otherwise the longer among end / noise / normal.
    @param durations: Array of rhythm durations
    @param labels: Array of rhythm labels.
    @return: Rhythm label as an integer.
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


def get_beat_label(labels):
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


def get_heart_rate_label(qrs_indices, fs=None) -> int:
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


def normalize(
    array: npt.ArrayLike, local: bool = True, filter_enable: bool = False
) -> npt.ArrayLike:
    """Normalize an array using the mean and standard deviation calculated over the entire dataset.

    Args:
        array (npt.ArrayLike):  Numpy array to normalize
        inplace (bool, optional): Whether to perform the normalization steps in-place. Defaults to False.
        local (bool, optional): Local mean and std or global. Defaults to True.
        filter_enable (bool, optional): Enable band-pass filter. Defaults to False.

    Returns:
        npt.ArrayLike: Normalized array
    """
    if filter_enable:
        filt_array = butter_bp_filter(
            array, lowcut=0.5, highcut=40, sample_rate=ds_sampling_rate, order=2
        )
    else:
        filt_array = np.copy(array)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        filt_array = sklearn.preprocessing.scale(
            filt_array, with_mean=True, with_std=True, copy=False
        )
    return filt_array


def _choose_random_segment(
    patients: npt.ArrayLike,
    size: Optional[int] = None,
    segment_p: Optional[Tuple[int, int, float]] = None,
):
    """Choose a random segment from an array of patient data. Each segment has the same probability of being chosen.
        Probability of a single segment may be changed by passing the `segment_p` argument.
    @param patients: An array of tuples of patient id and patient data.
    @param size: Number of the returned random segments. Defaults to 1 returned random segment.
    @param segment_p: Fixed probability of a chosen segment. `segment_p` should be a tuple of:
            (patient_index, segment_index, segment_probability)
    @return: One or more tuples (patient_index, segment_index) describing the randomly sampled
    segments from the patients buffer.
    Args:
        patients (_type_): _description_
        size (_type_, optional): _description_. Defaults to None.
        segment_p (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    num_segments_per_patient = np.array(
        [signal.shape[0] for _, (signal, _) in patients]
    )
    first_segment_index_by_patient = (
        np.cumsum(num_segments_per_patient) - num_segments_per_patient
    )
    num_segments = num_segments_per_patient.sum()
    if segment_p is None:
        p = np.ones(num_segments) / num_segments
    else:
        patient_index, segment_index, segment_prob = segment_p
        p_index = first_segment_index_by_patient[patient_index] + segment_index
        if num_segments <= p_index < 0:
            raise ValueError("The provided patient and segment indices are invalid")
        if 1.0 < segment_prob < 0.0:
            raise ValueError("Probability must lie in the [0, 1] interval")
        p = (1 - segment_prob) * np.ones(num_segments) / (num_segments - 1)
        p[p_index] = segment_prob
    segment_ids = np.random.choice(num_segments, size=size, p=p)
    if size is None:
        patient_index = (
            np.searchsorted(first_segment_index_by_patient, segment_ids, side="right")
            - 1
        )
        segment_index = segment_ids - first_segment_index_by_patient[patient_index]
        return patient_index, segment_index

    indices = []
    for segment_id in segment_ids:
        patient_index = (
            np.searchsorted(first_segment_index_by_patient, segment_id, side="right")
            - 1
        )
        segment_index = segment_id - first_segment_index_by_patient[patient_index]
        indices.append((patient_index, segment_index))
    return indices


def get_rhythm_statistics(
    db_path: str,
    patient_ids: Optional[npt.ArrayLike] = None,
    save_path: Optional[str] = None,
):
    """Utility function to extract rhythm statics across entire dataset. Useful for EDA.

    Args:
        db_path (str): Database path containing HDF5 files
        patient_ids (Optional[npt.ArrayLike], optional): Patients IDs to include. Defaults to all.
        save_path (Optional[str], optional): Parquet file path to save results. Defaults to None.

    Returns:
        pd.DataFrame: DataFrame of statistics
    """

    if patient_ids is None:
        patient_ids = get_patient_ids()
    pt_gen = uniform_patient_generator(
        db_path=db_path, patient_ids=patient_ids, repeat=False
    )
    stats = []
    for pt, segments in pt_gen:  # , total=len(patient_ids):
        # Group patient rhythms by type (segment, start, stop)
        segment_label_map: Dict[str, List[Tuple[str, int, int]]] = {}
        for seg_key, segment in segments.items():
            rlabels = segment["rlabels"][:]
            if rlabels.shape[0] == 0:
                continue  # Segment has no rhythm labels
            rlabels = rlabels[np.where(rlabels[:, 1] != IcentiaRhythm.noise.value)[0]]
            for i, l in enumerate(rlabels[::2, 1]):
                if l in (IcentiaRhythm.normal, IcentiaRhythm.afib, IcentiaRhythm.aflut):
                    rhy_start, rhy_stop = rlabels[i * 2 + 0, 0], rlabels[i * 2 + 1, 0]
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


def convert_dataset_pt_zip_to_hdf5(
    patient: int, zip_path: str, db_path: str, force: bool = False
):
    """Extract patient data from Icentia zipfile. Pulls out ECG data along with all labels.

    Args:
        patient (int): Patient id
        zip_path (str): Zipfile path
        db_path (str): Destination DB folder path
        force (bool, optional): Whether to override destination if it exists. Defaults to False.
    """
    import re  # pylint: disable=import-outside-toplevel

    import wfdb  # pylint: disable=import-outside-toplevel

    logger.info(f"Processing patient {patient}")
    pt_id = f"p{patient:05d}"
    pt_path = os.path.join(db_path, f"{pt_id}.h5")
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
            pt_seg_path = os.path.join(
                "/",
                os.path.splitext(os.path.basename(zp_rec_name))[0].replace("_", "/"),
            )
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
                name=os.path.join(pt_seg_path, "data"),
                data=data,
                compression="gzip",
                compression_opts=3,
            )
            h5.create_dataset(name=os.path.join(pt_seg_path, "blabels"), data=blabels)
            h5.create_dataset(name=os.path.join(pt_seg_path, "rlabels"), data=rlabels)
        except Exception as err:  # pylint: disable=broad-except
            print(f"Failed processing {zp_rec_name}", err)
            continue
    h5.close()


def convert_dataset_zip_to_hdf5(
    zip_path: str,
    db_path: str,
    patient_ids: Optional[npt.ArrayLike] = None,
    force: bool = False,
    num_workers: Optional[int] = None,
):
    """Convert zipped Icentia dataset into individial patient HDF5 files.

    Args:
        zip_path (str): Zipfile path
        db_path (str): Destination DB path
        patient_ids (Optional[npt.ArrayLike], optional): List of patient IDs to extract. Defaults to all.
        force (bool, optional): Whether to force re-download if destination exists. Defaults to False.
        num_workers (int, optional): # parallel workers. Defaults to os.cpu_count().
    """
    if not patient_ids:
        patient_ids = get_patient_ids()
    f = functools.partial(
        convert_dataset_pt_zip_to_hdf5, zip_path=zip_path, db_path=db_path, force=force
    )
    with Pool(processes=num_workers) as pool:
        _ = list(tqdm(pool.imap(f, patient_ids), total=len(patient_ids)))
