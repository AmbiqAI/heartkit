import os
import random
import zipfile
import tempfile
import dataclasses
from enum import IntEnum
from multiprocessing import Pool
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
import h5py
import numpy as np
import numpy.typing as npt
import sklearn.model_selection
from ..utils import EcgTask, filter_ecg_signal

# Patients containing AFIB events
afib_patients = [
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

class IcentiaRhythm(IntEnum):
    noise=0
    normal=1
    afib=2
    aflut=3
    end=4
    unknown=5

    @classmethod
    def hi_priority(cls) -> List[int]:
        return [cls.afib.value, cls.aflut.value]

    @classmethod
    def lo_priority(cls) -> List[int]:
        return [cls.noise.value, cls.normal.value, cls.end.value, cls.unknown.value]

class IcentiaBeat(IntEnum):
    undefined=0
    normal=1
    pac=2
    aberrated=3
    pvc=4

    @classmethod
    @property
    def hi_priority(cls) -> List[int]:
        return [cls.pac.value, cls.aberrated.value, cls.pvc.value]

    @classmethod
    @property
    def lo_priority(cls) -> List[int]:
        return [cls.undefined.value, cls.normal.value]

class IcentiaHeartRate(IntEnum):
    tachycardia=0
    bradycardia=1
    normal=2
    noise=3

ds_beat_names =  {b.name: b.value for b in IcentiaBeat}
ds_rhythm_names = {r.name: r.value for r in IcentiaRhythm}
ds_rhythm_map = { IcentiaRhythm.normal.value: 0, IcentiaRhythm.afib.value: 1 }

@dataclasses.dataclass
class IcentiaStats:
    patient_ids: npt.NDArray = dataclasses.field(default=np.arange(11000))
    sampling_rate: int = 250
    mean = 0.0018 # mean over entire dataset
    std = 1.3711 # std over entire dataset

ds_patient_ids = np.arange(10_000) # Reserve last 1000 for test only
ds_test_patient_ids = np.arange(10_000, 11_000)
ds_sampling_rate = 250 # Hz
ds_mean = 0.0018  # mean over entire dataset
ds_std = 1.3711 # std over entire dataset

ds_hr_names = {
    IcentiaHeartRate.tachycardia.value: 'tachy',
    IcentiaHeartRate.bradycardia.value: 'brady',
    IcentiaHeartRate.normal.value: 'normal',
    IcentiaHeartRate.noise.value: 'noise'
}

def train_test_split_patients(patient_ids: npt.NDArray, test_size: float, task: EcgTask = EcgTask.rhythm):
    if task == EcgTask.rhythm:
        afib_pt_ids = np.intersect1d(np.array(afib_patients), patient_ids)
        norm_pt_ids = np.setdiff1d(patient_ids, afib_pt_ids)
        norm_train_pt_ids, norm_val_pt_ids = sklearn.model_selection.train_test_split(norm_pt_ids, test_size=test_size)
        afib_train_pt_ids, afib_val_pt_ids = sklearn.model_selection.train_test_split(afib_pt_ids, test_size=test_size)
        train_pt_ids = np.concatenate((norm_train_pt_ids, afib_train_pt_ids))
        val_pt_ids = np.concatenate((norm_val_pt_ids, afib_val_pt_ids))
        np.random.shuffle(train_pt_ids)
        np.random.shuffle(val_pt_ids)
        return train_pt_ids, val_pt_ids
    # END IF
    return sklearn.model_selection.train_test_split(patient_ids, test_size=test_size)

def get_dataset_statistics(db_path: str, save_path: str):
    import pandas as pd
    pt_gen = uniform_patient_generator(db_path=db_path, patient_ids=ds_patient_ids, repeat=False)
    stats = []
    for pt, segments in tqdm(pt_gen, total=len(ds_patient_ids)):
        # Group patient rhythms by type (segment, start, stop)
        segment_label_map: Dict[str, List[Tuple[str, int, int]]] = {}
        for seg_key, segment in segments.items():
            rlabels = segment['rlabels'][:]
            if rlabels.shape[0] == 0:
                continue # Segment has no rhythm labels
            rlabels = rlabels[rlabels != 0]
            for i, l in enumerate(rlabels[::2,1]):
                if l in [IcentiaRhythm.normal, IcentiaRhythm.afib, IcentiaRhythm.aflut]:
                    rhy_start, rhy_stop = rlabels[i*2+0,0], rlabels[i*2+1,0]
                    stats.append(dict(pt=pt, rc=seg_key, rhythm=l, start=rhy_start, stop=rhy_stop, dur=rhy_stop-rhy_start))
                    segment_label_map[l] = segment_label_map.get(l, []) + [(seg_key, rlabels[i*2+0,0], rlabels[i*2+1,0])]
                # END IF
            # END FOR
        # END FOR
    # END FOR
    df = pd.DataFrame(stats)
    df.to_parquet(save_path)

def rhythm_data_generator(patient_generator, frame_size: int = 2048, samples_per_patient: int = 1):
    """
    Generate a stream of short signals and their corresponding rhythm label. These short signals are uniformly sampled
    from the segments in patient data by placing a frame in a random location within one of the segments.
    The corresponding label is then determined based on the rhythm durations within this frame.
    @param patient_generator: Generator that yields a tuple of patient id and patient data at each iteration.
    @param frame_size: Size of the frame that contains a short signal.
    @param samples_per_patient: Number of samples from one patient before new patient is pulled from the generator.
            This is done in order to decrease the number of i/o operations.
    @return: Generator of: input data of shape (frame_size, 1), output data as the corresponding rhythm label.
    """
    for pt, segments in patient_generator:
        print('.', end='')

        # Group patient rhythms by type (segment, start, stop)
        seg_label_map: Dict[str, List[Tuple[str, int, int]]] = {}
        for seg_key, segment in segments.items():
            rlabels = segment['rlabels'][:]
            if rlabels.shape[0] == 0:
                continue # Segment has no rhythm labels
            rlabels = rlabels[np.where(rlabels[:, 1] != 0)[0]]
            for i, l in enumerate(rlabels[::2,1]):
                xs, xe = rlabels[i*2+0,0], rlabels[i*2+1,0]
                seg_frame_size = xe - xs + 1
                if l in [IcentiaRhythm.normal, IcentiaRhythm.afib] and seg_frame_size > frame_size:
                    seg_label_map[l] = seg_label_map.get(l, []) + [(seg_key, xs, xe)]
                # END IF
            # END FOR
        # END FOR

        seg_samples: List[Tuple[str, int, int, int]] = []

        # Grab all arrhythmia instances
        afib_segments = seg_label_map.get(IcentiaRhythm.afib, [])
        for seg_key, rhy_start, rhy_end in afib_segments:
            for frame_start in range(rhy_start, rhy_end - frame_size + 1, frame_size):
                frame_end = frame_start + frame_size
                seg_samples.append((seg_key, frame_start, frame_end, ds_rhythm_map[IcentiaRhythm.afib]))
            # END FOR
        # END FOR

        # Grab normal instances
        norm_segments = seg_label_map.get(IcentiaRhythm.normal, [])
        while len(seg_samples) < samples_per_patient and norm_segments:
            seg_key, rhy_start, rhy_end = random.choice(norm_segments)
            frame_start = np.random.randint(rhy_start, rhy_end - frame_size + 1)
            frame_end = frame_start + frame_size
            seg_samples.append((seg_key, frame_start, frame_end, ds_rhythm_map[IcentiaRhythm.normal]))

        # Shuffle frames
        random.shuffle(seg_samples)

        # Generator
        for seg_key, frame_start, frame_end, label in seg_samples:
            x: npt.NDArray = segments[seg_key]['data'][frame_start:frame_end]
            yield x, label
        # END FOR
    # END FOR

def rhythm_data_generator_v1(patient_generator, frame_size: int = 2048, samples_per_patient: int = 1):
    """
    Generate a stream of short signals and their corresponding rhythm label. These short signals are uniformly sampled
    from the segments in patient data by placing a frame in a random location within one of the segments.
    The corresponding label is then determined based on the rhythm durations within this frame.
    @param patient_generator: Generator that yields a tuple of patient id and patient data at each iteration.
    @param frame_size: Size of the frame that contains a short signal.
    @param samples_per_patient: Number of samples from one patient before new patient is pulled from the generator.
            This is done in order to decrease the number of i/o operations.
    @return: Generator of: input data of shape (frame_size, 1), output data as the corresponding rhythm label.
    """
    for _, segments in patient_generator:
        for _ in range(samples_per_patient):
            segment = segments[np.random.choice(list(segments.keys()))]
            segment_size: int = segment['data'].shape[0]
            frame_start = np.random.randint(segment_size - frame_size)
            frame_end = frame_start + frame_size
            if segment['rlabels'].shape[0] == 0:
                continue
            x: npt.NDArray = segment['data'][frame_start:frame_end]
            # calculate the durations of each rhythm in the frame and determine the final label
            rhythm_bounds, rhythm_labels = segment['rlabels'][:, 0], segment['rlabels'][:, 1]
            frame_rhythm_durations, frame_rhythm_labels = get_rhythm_durations(
                rhythm_bounds, rhythm_labels, frame_start, frame_end
            )
            y = get_rhythm_label(frame_rhythm_durations, frame_rhythm_labels)
            yield x, y
        # END FOR
    # END FOR


def beat_data_generator(patient_generator, frame_size: int = 2048, samples_per_patient: int = 1):
    """
    Generate a stream of short signals and their corresponding beat label. These short signals are uniformly sampled
    from the segments in patient data by placing a frame in a random location within one of the segments.
    The corresponding label is then determined based on the beats within this frame.
    @param patient_generator: Generator that yields a tuple of patient id and patient data at each iteration.
    @param frame_size: Size of the frame that contains a short signal.
    @param samples_per_patient: Number of samples from one patient before new patient is pulled from the generator.
            This is done in order to decrease the number of i/o operations.
    @return: Generator of: input data of shape (frame_size, 1), output data as the corresponding beat label.
    """
    for _, segments in patient_generator:
        for _ in range(samples_per_patient):
            segment = segments[np.random.choice(list(segments.keys()))]
            segment_size: int = segment['data'].shape[0]
            frame_start = np.random.randint(segment_size - frame_size)
            frame_end = frame_start + frame_size
            if segment['blabels'].shape[0] == 0:
                continue
            x = segment['data'][frame_start:frame_end]
            beat_indices, beat_labels = segment['blabels'][:, 0], segment['blabels'][:, 1]
            # calculate the count of each beat type in the frame and determine the final label
            _, frame_beat_labels = get_complete_beats(beat_indices, beat_labels, frame_start, frame_end)
            y = get_beat_label(frame_beat_labels)
            yield x, y
        # END FOR
    # END FOR


def heart_rate_data_generator(patient_generator, frame_size=2048, label_frame_size=None, samples_per_patient=1):
    """
    Generate a stream of short signals and their corresponding heart rate label. These short signals are uniformly
    sampled from the segments in patient data by placing a frame in a random location within one of the segments.
    The corresponding label is then determined based on the beats within this frame.
    @param patient_generator: Generator that yields a tuple of patient id and patient data at each iteration.
    @param frame_size: Size of the frame that contains a short input signal.
    @param label_frame_size: Size of the frame centered on the input signal frame, that contains a short signal used
            for determining the label. By default equal to the size of the input signal frame.
    @param samples_per_patient: Number of samples from one patient before new patient is pulled from the generator.
            This is done in order to decrease the number of i/o operations.
    @return: Generator of: input data of shape (frame_size, 1),
    output data as the corresponding heart rate label.
    """
    if label_frame_size is None:
        label_frame_size = frame_size
    max_frame_size = max(frame_size, label_frame_size)
    for _, segments in patient_generator:
        for _ in range(samples_per_patient):
            segment = segments[np.random.choice(list(segments.keys()))]
            segment_size: int = segment['data'].shape[0]
            frame_center = np.random.randint(segment_size - max_frame_size) + max_frame_size // 2
            signal_frame_start = frame_center - frame_size // 2
            signal_frame_end = frame_center + frame_size // 2
            x = segment['data'][signal_frame_start:signal_frame_end]
            label_frame_start = frame_center - label_frame_size // 2
            label_frame_end = frame_center + label_frame_size // 2
            beat_indices = segment['blabels'][:, 0]
            frame_beat_indices = get_complete_beats(beat_indices, start=label_frame_start, end=label_frame_end)
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
            # randomly choose a frame that lies within the segment i.e. no zero-padding is necessary
            segment = segments[np.random.choice(list(segments.keys()))]
            segment_size = segment['data'].shape[0]
            frame_start = np.random.randint(segment_size - frame_size)
            frame_end = frame_start + frame_size
            x = segment['data'][frame_start:frame_end]
            # x = np.expand_dims(x, axis=1)  # add channel dimension
            yield x
        # END FOR
    # END FOR

def uniform_patient_generator(
        db_path: str,
        patient_ids: List[int],
        repeat: bool = True,
        shuffle: bool = True,
    ):
    """
    Yield data for each patient in the array.
    @param db_path: Database path.
    @param patient_ids: Array of patient ids.
    @param repeat: Whether to restart the generator when the end of patient array is reached.
    @param shuffle: Whether to shuffle patient ids.
    @return: Generator that yields a tuple of patient id and patient data.
    """
    if shuffle:
        patient_ids = np.copy(patient_ids)
        while True:
            if shuffle:
                np.random.shuffle(patient_ids)
                print('x', end='')
            for patient_id in patient_ids:
                pt_key = f'p{patient_id:05d}'
                with h5py.File(os.path.join(db_path, f'{pt_key}.h5'), mode='r') as h5:
                    patient_data = h5[pt_key]
                    yield patient_id, patient_data
            # END FOR
            if not repeat:
                break
        # END WHILE
    # END WITH


def random_patient_generator(
        db_path: str,
        patient_ids: List[int],
        patient_weights: Optional[List[int]] = None,
    ):
    """
    Samples patient data from the provided patient distribution.
    @param db_path: Database path.
    @param patient_ids: Array of patient ids.
    @param patient_weights: Probabilities associated with each patient. By default assumes a uniform distribution.
    @return: Generator that yields a tuple of patient id and patient data.
    """
    with h5py.File(db_path, mode='r') as h5:
        while True:
            # NOTE: Currently only 1024 patients are selected w/ repeats
            for patient_id in np.random.choice(patient_ids, size=1024, p=patient_weights):
                patient_data = h5[f'p{patient_id:05d}']
                yield patient_id, patient_data


def count_labels(labels: List[Tuple[int, int]], num_classes: int):
    """
    Count the number of labels in all segments.
    @param labels: Array of tuples of indices, labels. Each tuple contains the labels within a segment.
    @param num_classes: Number of classes (either beat or rhythm depending on the label type).
    @return: Numpy array of label counts of shape (num_segments, num_classes).
    """
    return np.array([
        np.bincount(segment_labels, minlength=num_classes) for _, segment_labels in labels
    ])


def calculate_durations(labels, num_classes):
    """
    Calculate the duration of each label in all segments.
    @param labels: Array of tuples of indices, labels. Each tuple corresponds to a segment.
    @param num_classes: Number of classes (either beat or rhythm depending on the label type).
    @return: Numpy array of label durations of shape (num_segments, num_classes).
    """
    num_segments = len(labels)
    durations = np.zeros((num_segments, num_classes), dtype='int32')
    for segment_index, (segment_indices, segment_labels) in enumerate(labels):
        segment_durations = np.diff(segment_indices, prepend=0)
        for label in range(num_classes):
            durations[segment_index, label] = segment_durations[segment_labels == label].sum()
    return durations

def flatten_raw_labels(raw_labels):
    """
    Flatten raw labels from a patient file for easier processing.
    @param raw_labels: Array of dictionaries containing the beat and rhythm labels for each segment.
            Note, that beat and rhythm label indices do not always overlap.
    @return: Dictionary of beat and rhythm arrays.
    Each array contains a tuple of indices, labels for each segment.
    """
    num_segments = len(raw_labels)
    labels = {'btype': [], 'rtype': [], 'size': num_segments}
    for label_type in ['btype', 'rtype']:
        for segment_labels in raw_labels:
            flat_indices = []
            flat_labels = []
            for label, indices in enumerate(segment_labels[label_type]):
                flat_indices.append(indices)
                flat_labels.append(np.repeat(label, len(indices)))
            flat_indices = np.concatenate(flat_indices)
            flat_labels = np.concatenate(flat_labels)
            sort_index = np.argsort(flat_indices)
            flat_indices = flat_indices[sort_index]
            flat_labels = flat_labels[sort_index]
            labels[label_type].append((flat_indices, flat_labels))
    return labels


def get_rhythm_durations(indices, labels=None, start=0, end=None):
    """
    Compute the durations of each rhythm within the specified frame.
    The indices are assumed to specify the end of a rhythm.
    @param indices: Array of rhythm indices. Indices are assumed to be sorted.
    @param labels: Array of rhythm labels.
    @param start: Index of the first sample in the frame.
    @param end: Index of the last sample in the frame. By default the last element in the indices array.
    @return: Tuple of: (rhythm durations, rhythm labels) in the provided frame
    or only rhythm durations if labels are not provided.
    """
    indices = indices.copy()
    labels = labels.copy()
    keep_indices = np.where(labels%5 != 0)[0]
    indices = indices[keep_indices]
    labels = labels[keep_indices]
    if end is None:
        end = indices[-1]
    if start >= end:
        raise ValueError('`end` must be greater than `start`')
    # find the first rhythm label after the beginning of the frame
    start_index = np.searchsorted(indices, start, side='left')
    if start_index%2 == 1:
        indices[start_index-1] = start
        start_index -= 1

    # find the first rhythm label after or exactly at the end of the frame
    end_index = np.searchsorted(indices, end, side='left')
    if end >= indices[-1]:
        end_index = len(indices)
    elif end_index%2 == 1:
        indices[end_index] = end
        end_index += 1

    frame_indices = indices[start_index:end_index]
    # compute the duration of each rhythm adjusted for the beginning and end of the frame
    frame_rhythm_durations = np.diff(frame_indices.reshape((-1,2))).reshape((-1))
    total_duration = end - start
    unknown_duration = total_duration - frame_rhythm_durations.sum()
    if labels is None:
        return frame_rhythm_durations
    else:
        frame_labels = labels[start_index:end_index][::2]
        frame_rhythm_durations = np.concatenate((frame_rhythm_durations, [unknown_duration]))
        frame_labels = np.concatenate((frame_labels, [IcentiaRhythm.unknown.value]))
        return frame_rhythm_durations, frame_labels
    # END IF


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
        raise ValueError('`end` must be greater than `start`')
    start_index = np.searchsorted(indices, start, side='left') + 1
    end_index = np.searchsorted(indices, end, side='right')
    indices_slice = indices[start_index:end_index]
    if labels is None:
        return indices_slice
    else:
        label_slice = labels[start_index:end_index]
        return indices_slice, label_slice


def get_rhythm_label(durations, labels):
    """
    Determine rhythm label based on the longest rhythm among undefined / afib / aflut if present,
    otherwise the longer among end / noise / normal.
    @param durations: Array of rhythm durations
    @param labels: Array of rhythm labels.
    @return: Rhythm label as an integer.
    """
    # sum up the durations of each rhythm
    summed_durations = np.zeros(len(ds_rhythm_names))
    for rhythm in IcentiaRhythm:
        summed_durations[rhythm.value] = durations[labels == rhythm.value].sum()
    longest_hp_rhythm = np.argmax(summed_durations[IcentiaRhythm.hi_priority()])
    if summed_durations[IcentiaRhythm.hi_priority()][longest_hp_rhythm] > 0:
        y = IcentiaRhythm.hi_priority()[longest_hp_rhythm]
    else:
        longest_lp_rhythm = np.argmax(summed_durations[IcentiaRhythm.lo_priority()])
        # handle the case of no detected rhythm
        if summed_durations[IcentiaRhythm.lo_priority()][longest_lp_rhythm] > 0:
            y = IcentiaRhythm.lo_priority()[longest_lp_rhythm]
        else:
            y = 0  # undefined rhythm
    return y


def get_beat_label(labels):
    """
    Determine beat label based on the occurrence of pac / abberated / pvc,
    otherwise pick the most common beat type among the normal / undefined.
    @param labels: Array of beat labels.
    @return: Beat label as an integer.
    """
    # calculate the count of each beat type in the frame
    beat_counts = np.bincount(labels, minlength=len(ds_beat_names))
    most_hp_beats = np.argmax(beat_counts[IcentiaBeat.hi_priority()])
    if beat_counts[IcentiaBeat.hi_priority()][most_hp_beats] > 0:
        y = IcentiaBeat.hi_priority()[most_hp_beats]
    else:
        most_lp_beats = np.argmax(beat_counts[IcentiaBeat.lo_priority()])
        # handle the case of no detected beats
        if beat_counts[IcentiaBeat.lo_priority()][most_lp_beats] > 0:
            y = IcentiaBeat.lo_priority()[most_lp_beats]
        else:
            y = 0  # undefined beat
    return y


def get_heart_rate_label(qrs_indices, fs=None):
    """
    Determine the heart rate label based on an array of QRS indices (separating individual heartbeats).
    The QRS indices are assumed to be measured in seconds if sampling frequency `fs` is not specified.
    The heartbeat label is based on the following BPM (beats per minute) values: (0) tachycardia <60 BPM,
    (1) bradycardia >100 BPM, (2) healthy 60-100 BPM, (3) noisy if QRS detection failed.
    @param qrs_indices: Array of QRS indices.
    @param fs: Sampling frequency of the signal.
    @return: Heart rate label as an integer.
    """
    if len(qrs_indices) > 1:
        rr_intervals = np.diff(qrs_indices)
        if fs is not None:
            rr_intervals = rr_intervals / fs
        bpm = 60 / rr_intervals.mean()
        if bpm < 60:
            return IcentiaHeartRate.bradycardia.value
        elif bpm <= 100:
            return IcentiaHeartRate.normal.value
        else:
            return IcentiaHeartRate.tachycardia.value
    else:
        return IcentiaHeartRate.noise.value


def normalize(array: npt.NDArray, inplace=False, local=True):
    """
    Normalize an array using the mean and standard deviation calculated over the entire dataset.
    @param array: Numpy array to normalize.
    @param inplace: Whether to perform the normalization steps in-place.
    @return: Normalized array.
    """
    filt_array = array
    # filt_array = filter_ecg_signal(array, lowcut=0.5, highcut=30, sample_rate=ds_sampling_rate, order=2)
    mu = np.mean(filt_array) if local else ds_mean
    std = np.std(filt_array) if local else ds_std

    filt_array = (filt_array - mu) / std
    if inplace:
        array[:] = filt_array
    return filt_array


def _choose_random_segment(patients, size=None, segment_p=None):
    """
    Choose a random segment from an array of patient data. Each segment has the same probability of being chosen.
    Probability of a single segment may be changed by passing the `segment_p` argument.
    @param patients: An array of tuples of patient id and patient data.
    @param size: Number of the returned random segments. Defaults to 1 returned random segment.
    @param segment_p: Fixed probability of a chosen segment. `segment_p` should be a tuple of:
            (patient_index, segment_index, segment_probability)
    @return: One or more tuples (patient_index, segment_index) describing the randomly sampled
    segments from the patients buffer.
    """
    num_segments_per_patient = np.array([signal.shape[0] for _, (signal, _) in patients])
    first_segment_index_by_patient = np.cumsum(num_segments_per_patient) - num_segments_per_patient
    num_segments = num_segments_per_patient.sum()
    if segment_p is None:
        p = np.ones(num_segments) / num_segments
    else:
        patient_index, segment_index, segment_prob = segment_p
        p_index = first_segment_index_by_patient[patient_index] + segment_index
        if num_segments <= p_index < 0:
            raise ValueError('The provided patient and segment indices are invalid')
        if 1. < segment_prob < 0.:
            raise ValueError('Probability must lie in the [0, 1] interval')
        p = (1 - segment_prob) * np.ones(num_segments) / (num_segments - 1)
        p[p_index] = segment_prob
    segment_ids = np.random.choice(num_segments, size=size, p=p)
    if size is None:
        patient_index = np.searchsorted(first_segment_index_by_patient, segment_ids, side='right') - 1
        segment_index = segment_ids - first_segment_index_by_patient[patient_index]
        return patient_index, segment_index
    else:
        indices = []
        for segment_id in segment_ids:
            patient_index = np.searchsorted(first_segment_index_by_patient, segment_id, side='right') - 1
            segment_index = segment_id - first_segment_index_by_patient[patient_index]
            indices.append((patient_index, segment_index))
        return indices

def convert_pt_zip_to_hdf5(patient: int, zip_path: str, h5_path: str):
    import wfdb
    import re
    print(f'Processing patient {patient}')
    sym_map = {'Q': 0, 'N': 1, 'S': 2, 'a': 3, 'V': 4}
    rhy_map = {'': 0, '(N': 1, '(AFIB': 2, '(AFL': 3, ')': 4}

    pt_id = f'p{patient:05d}'
    zp = zipfile.ZipFile(zip_path, mode='r')
    h5 = h5py.File(os.path.join(h5_path, f'{pt_id}.h5'), mode='w')

    # Find all patient .dat file indices
    zp_rec_names = filter(lambda f: re.match(f'{pt_id}_[A-z0-9]+.dat', os.path.basename(f)), (f.filename for f in zp.filelist))
    for zp_rec_name in zp_rec_names:
        try:
            zp_hdr_name = zp_rec_name.replace('.dat', '.hea')
            zp_atr_name = zp_rec_name.replace('.dat', '.atr')

            with tempfile.TemporaryDirectory() as tmpdir:
                rec_fpath = os.path.join(tmpdir, os.path.basename(zp_rec_name))
                atr_fpath = rec_fpath.replace('.dat', '.atr')
                hdr_fpath = rec_fpath.replace('.dat', '.hea')
                with open(hdr_fpath, 'wb') as fp:
                    fp.write(zp.read(zp_hdr_name))
                with open(rec_fpath, 'wb') as fp:
                    fp.write(zp.read(zp_rec_name))
                with open(atr_fpath, 'wb') as fp:
                    fp.write(zp.read(zp_atr_name))
                rec = wfdb.rdrecord(os.path.splitext(rec_fpath)[0], physical=True)
                atr = wfdb.rdann(os.path.splitext(atr_fpath)[0], extension='atr')
            pt_seg_path = os.path.join('/', os.path.splitext(os.path.basename(zp_rec_name))[0].replace('_', '/'))
            data = rec.p_signal.astype(np.float16)
            blabels = np.array([[atr.sample[i], sym_map.get(s)] for i,s in enumerate(atr.symbol) if s in sym_map], dtype=np.int32)
            rlabels = np.array([[atr.sample[i], rhy_map.get(atr.aux_note[i], 0)] for i,s in enumerate(atr.symbol) if s == '+'], dtype=np.int32)
            h5.create_dataset(name=os.path.join(pt_seg_path, 'data'), data=data, compression="gzip", compression_opts=3)
            h5.create_dataset(name=os.path.join(pt_seg_path, 'blabels'), data=blabels)
            h5.create_dataset(name=os.path.join(pt_seg_path, 'rlabels'), data=rlabels)
        except Exception as err:
            print(f'Failed processing {zp_rec_name}', err)
            continue
    h5.close()

def convert_zip_to_hdf5(zip_path: str, h5_path: str):
    import functools
    f = functools.partial(convert_pt_zip_to_hdf5, zip_path=zip_path, h5_path=h5_path)
    with Pool(processes=10) as pool:
        print('Started')
        pool.map(f, ds_patient_ids)
        print('Finished')
