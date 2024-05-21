import random
from typing import Generator, Iterable

import numpy as np
import numpy.typing as npt
import physiokit as pk

from ....datasets.defines import PatientGenerator
from ....datasets.icentia11k import IcentiaBeat, IcentiaDataset
from ..defines import HKBeat

IcentiaBeatMap = {
    IcentiaBeat.undefined: HKBeat.noise,
    IcentiaBeat.normal: HKBeat.normal,
    IcentiaBeat.pac: HKBeat.pac,
    IcentiaBeat.aberrated: HKBeat.pac,
    IcentiaBeat.pvc: HKBeat.pvc,
}


# Filter beats based on neighboring beats
def beat_filter_func(i: int, blabels: npt.NDArray, beat: IcentiaBeat):
    """Filter beats based on neighboring beats"""
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


def icentia11k_label_map(
    label_map: dict[int, int] | None = None,
) -> dict[int, int]:
    """Get label map

    Args:
        label_map (dict[int, int]|None): Label map

    Returns:
        dict[int, int]: Label map
    """
    return {k: label_map.get(v, -1) for (k, v) in IcentiaBeatMap.items()}


def icentia11k_data_generator(
    patient_generator: PatientGenerator,
    ds: IcentiaDataset,
    frame_size: int,
    samples_per_patient: int | list[int] = 1,
    target_rate: int | None = None,
    label_map: dict[int, int] | None = None,
    label_type: str = "beat",
) -> Generator[tuple[npt.NDArray, int], None, None]:
    """Generate frames w/ rhythm labels (e.g. afib) using patient generator.

    Args:
        patient_generator (PatientGenerator): Patient Generator
        ds: IcentiaDataset
        frame_size (int): Frame size
        samples_per_patient (int | list[int], optional): # samples per patient. Defaults to 1.
        target_rate (int|None, optional): Target rate. Defaults to None.
        label_map (dict[int, int] | None, optional): Label map. Defaults to None.
        label_type (str, optional): Label type. Defaults to "beat".

    Returns:
        Generator[tuple[npt.NDArray, int], None, None]: Sample generator
    """
    if target_rate is None:
        target_rate = ds.sampling_rate
    # END IF

    nlabel_threshold = 0.25
    blabel_padding = 20

    # Target labels and mapping
    tgt_labels = sorted(list(set((lbl for lbl in label_map.values() if lbl != -1))))
    label_key = ds.label_key(label_type)

    tgt_map = icentia11k_label_map(label_map=label_map)
    num_classes = len(tgt_labels)

    # If samples_per_patient is a list, then it must be the same length as nclasses
    if isinstance(samples_per_patient, Iterable):
        samples_per_tgt = samples_per_patient
    else:
        num_per_tgt = int(max(1, samples_per_patient / num_classes))
        samples_per_tgt = num_per_tgt * [num_classes]

    input_size = int(np.round((ds.sampling_rate / target_rate) * frame_size))

    # For each patient
    for pt in patient_generator:
        with ds.patient_data(pt) as segments:
            # This maps segment index to segment key
            seg_map: list[str] = list(segments.keys())

            # Capture beat locations for each segment
            pt_beat_map = [[] for _ in range(num_classes)]
            for seg_idx, seg_key in enumerate(seg_map):
                # Get beat labels
                blabels = segments[seg_key][label_key][:]

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
                    # fn = functools.partial(beat_filter_func, blabels=blabels, beat=beat)
                    # beat_idxs = filter(fn, beat_idxs)
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
                x = np.nan_to_num(data[frame_start:frame_end]).astype(np.float32)
                if ds.sampling_rate != target_rate:
                    x = pk.signal.resample_signal(x, ds.sampling_rate, target_rate, axis=0)
                y = beat
                yield x, y
            # END FOR
        # END WITH
    # END FOR
