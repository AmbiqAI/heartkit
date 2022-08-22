import os
import random
import zipfile
import tempfile
import dataclasses
from enum import IntEnum
from multiprocessing import Pool
from typing import Dict, List, Optional, Tuple
import h5py
import numpy as np
import numpy.typing as npt

class IcentiaRhythm(IntEnum):
    noise=0
    normal=1
    afib=2
    aflut=3
    end=4
    unknown=5

    @classmethod
    @property
    def hi_priority(cls) -> List[int]:
        return [cls.afib.value, cls.aflut.value]

    @classmethod
    @property
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
ds_rhythm_map = {IcentiaRhythm.normal.value: 0, IcentiaRhythm.afib.value: 1, IcentiaRhythm.aflut.value: 2}

@dataclasses.dataclass
class IcentiaStats:
    patient_ids: npt.NDArray = dataclasses.field(default=np.arange(11000))
    sampling_rate: int = 250
    mean = 0.0018 # mean over entire dataset
    std = 1.3711 # std over entire dataset

ds_patient_ids = np.arange(11000)
ds_sampling_rate = 250
ds_mean = 0.0018  # mean over entire dataset
ds_std = 1.3711   # std over entire dataset

ds_hr_names = {
    IcentiaHeartRate.tachycardia.value: 'tachy',
    IcentiaHeartRate.bradycardia.value: 'brady',
    IcentiaHeartRate.normal.value: 'normal',
    IcentiaHeartRate.noise.value: 'noise'
}

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

        # Group patient rhythms by type (segment, start, stop)
        segment_label_map: Dict[str, List[Tuple[str, int, int]]] = {}
        for seg_key, segment in segments.items():
            rlabels = segment['rlabels'][:]
            if rlabels.shape[0] == 0:
                continue # Segment has no rhythm labels
            for i, l in enumerate(rlabels[::2,1]):
                if l in [IcentiaRhythm.normal, IcentiaRhythm.afib, IcentiaRhythm.aflut]:
                    segment_label_map[l] = segment_label_map.get(l, []) + [(seg_key, rlabels[i*2+0,0], rlabels[i*2+1,0])]
                # END IF
            # END FOR
        # END FOR

        # Skip patient if no labels
        if not segment_label_map:
            continue

        num_samples = 0
        num_attempts = 0
        while num_samples < samples_per_patient:
            label = np.random.choice(list(segment_label_map.keys()))
            seg_key, rhy_start, rhy_end = random.choice(segment_label_map[label])
            segment = segments[seg_key]
            segment_size: int = segment['data'].shape[0]
            frame_start = max(min(np.random.randint(rhy_start, rhy_end), segment_size-frame_size), 0)
            frame_end = frame_start + frame_size
            x: npt.NDArray = segment['data'][frame_start:frame_end]
            # calculate the durations of each rhythm in the frame and determine the final label
            if segment['rlabels'].shape[0] == 0:
                continue
            rhythm_bounds, rhythm_labels = segment['rlabels'][:, 0], segment['rlabels'][:, 1]
            frame_rhythm_durations, frame_rhythm_labels = get_rhythm_durations(
                rhythm_bounds, rhythm_labels, frame_start, frame_end
            )
            y = get_rhythm_label(frame_rhythm_durations, frame_rhythm_labels)
            if y in ds_rhythm_map:
                num_samples += 1
                yield x, ds_rhythm_map[y]
            else:
                if num_attempts == samples_per_patient:
                    print(pt)
                    num_samples = samples_per_patient
                num_attempts += 1
        # END WHILE
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


def cpc_data_generator(buffered_patient_generator, context_size, ns, frame_size=2048, context_overlap=0,
                       positive_offset=0, ns_same_segment_prob=None, samples_per_patient=1):
    """
    Generate a stream of input, output data for the Contrastive Predictive Coding
    (Oord et al. (2018) https://arxiv.org/abs/1807.03748) representation learning approach.
    The idea is to predict the positive sample from the future among distractors (negative samples)
    based on some context. Note, that while we attempt to capture the idea of the proposed approach for 1D data,
    our implementation deviates from the implementation described in the paper.
    @param buffered_patient_generator: Generator that yields a buffer filled with patient data at each iteration.
            Patient data may contain only signals, since labels are not used.
    @param context_size: Number of frames that make up the context.
    @param ns: Number of negative samples.
    @param frame_size: Size of the frame that contains a short signal.
    @param context_overlap: Size of the overlap between two consecutive frames.
    @param positive_offset: Offset from the end of the context measured in frames, that describes the distance
            between the context and the positive sample. If the offset is 0 then the positive sample comes directly
            after the context.
    @param ns_same_segment_prob: Probability, that a negative sample will come from the same segment as
            the positive sample. If the probability is 0 then all negative samples are equally likely to come from
            every segment in the buffer. If the probability is 1 then all negative samples come from the same segment
            as the positive sample. By default, all segments have the same probability of being sampled.
    @param samples_per_patient: Number of samples before the buffer is updated.
            This is done in order to decrease the number of i/o operations.
    @return: Generator of: input data as a dictionary {context, samples}, output data as the position
    of the positive sample in the samples array. Context is an array of (optionally) overlapping frames.
    Samples is an array of frames to predict from (1 positive sample and rest negative samples).
    """
    # compute context size measured in amplitude samples, adjust for the frame overlap
    context_size = context_size * (frame_size - context_overlap)
    for patients_buffer in buffered_patient_generator:
        for _ in range(samples_per_patient):
            # collect (optionally) overlapping frames that will form the context
            # choose context start such that the positive sample will remain within the segment
            patient_index, segment_index = _choose_random_segment(patients_buffer)
            _, (signal, _) = patients_buffer[patient_index]
            segment_size = signal.shape[1]
            context_start = np.random.randint(segment_size - (context_size + frame_size * (positive_offset + 1)))
            context_end = context_start + context_size
            context = []
            for context_frame_start in range(context_start, context_end, frame_size - context_overlap):
                context_frame_end = context_frame_start + frame_size
                context_frame = signal[segment_index, context_frame_start:context_frame_end]
                context_frame = np.expand_dims(context_frame, axis=1)  # add channel dimension
                context.append(context_frame)
            context = np.array(context)
            # collect positive sample from the future relative to the context
            positive_sample_start = context_start + context_size + frame_size * positive_offset
            positive_sample_end = positive_sample_start + frame_size
            positive_sample = signal[segment_index, positive_sample_start:positive_sample_end]
            positive_sample = np.expand_dims(positive_sample, axis=1)  # add channel dimension
            # collect negative samples
            #  note that if the patient buffer contains only 1 patient then
            #  all negative samples will also come from this patient
            samples = []
            p = (patient_index, segment_index, ns_same_segment_prob) if ns_same_segment_prob else None
            ns_indices = _choose_random_segment(patients_buffer, size=ns, segment_p=p)
            for ns_patient_index, ns_segment_index in ns_indices:
                _, (ns_signal, _) = patients_buffer[ns_patient_index]
                ns_segment_size = ns_signal.shape[1]
                negative_sample_start = np.random.randint(ns_segment_size - frame_size)
                negative_sample_end = negative_sample_start + frame_size
                negative_sample = ns_signal[ns_segment_index, negative_sample_start:negative_sample_end]
                negative_sample = np.expand_dims(negative_sample, axis=1)  # add channel dimension
                samples.append(negative_sample)
            # randomly insert the positive sample among the negative samples
            # the label references the position of the positive sample among all samples
            y = np.random.randint(ns + 1)
            samples.insert(y, positive_sample)
            samples = np.array(samples)
            x = {'context': context, 'samples': samples}
            yield x, y
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
    longest_hp_rhythm = np.argmax(summed_durations[IcentiaRhythm.hi_priority])
    if summed_durations[IcentiaRhythm.hi_priority][longest_hp_rhythm] > 0:
        y = IcentiaRhythm.hi_priority[longest_hp_rhythm]
    else:
        longest_lp_rhythm = np.argmax(summed_durations[IcentiaRhythm.lo_priority])
        # handle the case of no detected rhythm
        if summed_durations[IcentiaRhythm.lo_priority][longest_lp_rhythm] > 0:
            y = IcentiaRhythm.lo_priority[longest_lp_rhythm]
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
    most_hp_beats = np.argmax(beat_counts[IcentiaBeat.hi_priority])
    if beat_counts[IcentiaBeat.hi_priority][most_hp_beats] > 0:
        y = IcentiaBeat.hi_priority[most_hp_beats]
    else:
        most_lp_beats = np.argmax(beat_counts[IcentiaBeat.lo_priority])
        # handle the case of no detected beats
        if beat_counts[IcentiaBeat.lo_priority][most_lp_beats] > 0:
            y = IcentiaBeat.lo_priority[most_lp_beats]
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


def normalize(array, inplace=False):
    """
    Normalize an array using the mean and standard deviation calculated over the entire dataset.
    @param array: Numpy array to normalize.
    @param inplace: Whether to perform the normalization steps in-place.
    @return: Normalized array.
    """
    if inplace:
        array -= ds_mean
        array /= ds_std
    else:
        array = (array - ds_mean) / ds_std
    return array


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


if __name__ == "__main__":
    zip_path = '/Users/adampage/Ambiq/ecg-arr/db/icentia11k/icentia11k.zip'
    h5_dir = '/Users/adampage/Ambiq/ecg-arr/db/icentia11k/patients'
    db_path = h5_dir

    # convert_zip_to_hdf5(zip_path=zip_path, h5_dir=h5_dir)

    pt_gen = uniform_patient_generator(db_path=db_path, patient_ids=ds_patient_ids, repeat=False)
    results = []
    ds = rhythm_data_generator(pt_gen, samples_per_patient=10*250)
    for sample in ds:
        results += [sample[1]]
    print(results)
