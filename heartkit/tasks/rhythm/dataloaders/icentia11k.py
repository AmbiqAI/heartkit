import random
from typing import Generator, Iterable

import numpy as np
import numpy.typing as npt
import physiokit as pk

from ....datasets.defines import PatientGenerator
from ....datasets.icentia11k import IcentiaDataset, IcentiaRhythm
from ..defines import HKRhythm

IcentiaRhythmMap = {
    IcentiaRhythm.noise: HKRhythm.noise,
    IcentiaRhythm.normal: HKRhythm.sr,
    IcentiaRhythm.afib: HKRhythm.afib,
    IcentiaRhythm.aflut: HKRhythm.aflut,
    IcentiaRhythm.end: HKRhythm.noise,
}


def icentia11k_label_map(
    label_map: dict[int, int] | None = None,
) -> dict[int, int]:
    """Get label map

    Args:
        label_map (dict[int, int]|None): Label map

    Returns:
        dict[int, int]: Label map
    """
    return {k: label_map.get(v, -1) for (k, v) in IcentiaRhythmMap.items()}


def icentia11k_data_generator(
    patient_generator: PatientGenerator,
    ds: IcentiaDataset,
    frame_size: int,
    samples_per_patient: int | list[int] = 1,
    target_rate: int | None = None,
    label_map: dict[int, int] | None = None,
) -> Generator[tuple[npt.NDArray, int], None, None]:
    """Generate frames w/ rhythm labels (e.g. afib) using patient generator.

    Args:
        patient_generator (PatientGenerator): Patient Generator
        ds: IcentiaDataset
        frame_size (int): Frame size
        samples_per_patient (int | list[int], optional): # samples per patient. Defaults to 1.
        target_rate (int|None, optional): Target rate. Defaults to None.
        label_map (dict[int, int] | None, optional): Label map. Defaults to None.

    Returns:
        Generator[tuple[npt.NDArray, int], None, None]: Sample generator
    """
    if target_rate is None:
        target_rate = ds.sampling_rate
    # END IF

    # Target labels and mapping
    tgt_labels = sorted(list(set((lbl for lbl in label_map.values() if lbl != -1))))
    tgt_map = icentia11k_label_map(label_map=label_map)
    label_key = ds.label_key("rhythm")
    num_classes = len(tgt_labels)

    # If samples_per_patient is a list, then it must be the same length as num_classes
    if isinstance(samples_per_patient, Iterable):
        samples_per_tgt = samples_per_patient
    else:
        num_per_tgt = int(max(1, samples_per_patient / num_classes))
        samples_per_tgt = num_per_tgt * [num_classes]
    # END IF

    input_size = int(np.round((ds.sampling_rate / target_rate) * frame_size))

    # Group patient rhythms by type (segment, start, stop, delta)
    for pt in patient_generator:

        with ds.patient_data(pt) as segments:
            # This maps segment index to segment key
            seg_map: list[str] = list(segments.keys())

            pt_tgt_seg_map = [[] for _ in tgt_labels]
            for seg_idx, seg_key in enumerate(seg_map):
                # Grab rhythm labels
                labels = segments[seg_key][label_key][:]

                # Skip if no labels
                if not labels.shape[0]:
                    continue
                labels = labels[np.where(labels[:, 1] != IcentiaRhythm.noise.value)[0]]
                if not labels.shape[0]:
                    continue

                # Unpack start, end, and label
                xs, xe, xl = labels[0::2, 0], labels[1::2, 0], labels[0::2, 1]

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
                if ds.sampling_rate != target_rate:
                    x = pk.signal.resample_signal(x, ds.sampling_rate, target_rate, axis=0)
                yield x, label
            # END FOR
        # END WITH
    # END FOR
