import copy
import random
import functools
from typing import Generator, Iterable

import numpy as np
import numpy.typing as npt
import physiokit as pk
import helia_edge as helia

from ....datasets import HKDataloader, IcentiaDataset, IcentiaBeat
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


class Icentia11kDataloader(HKDataloader):
    def __init__(self, ds: IcentiaDataset, **kwargs):
        """Icentia11k Dataloader for training beat tasks"""
        super().__init__(ds=ds, **kwargs)

        # Update label map
        if self.label_map:
            self.label_map = {k: self.label_map[v] for (k, v) in IcentiaBeatMap.items() if v in self.label_map}
        # END DEF
        self.label_type = "beat"
        # {PT: [label_idx: [segment, location]]}
        self._pts_beat_map: dict[str, list[npt.NDArray]] = {}

    def _create_beat_map(self, patient_id: int, enable_filter: bool = False):
        """On initial access, create beat map for patient to improve speed"""
        nlabel_threshold = 0.25
        blabel_padding = 20

        # Target labels and mapping
        tgt_labels = sorted(set(self.label_map.values()))
        label_key = self.ds.label_key(self.label_type)
        num_classes = len(tgt_labels)

        with self.ds.patient_data(patient_id) as segments:
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
                    beat_class = self.label_map.get(beat, -1)
                    if beat_class < 0 or beat_class >= num_classes:
                        continue

                    # Get all beat type indices
                    beat_idxs = np.where(blabels[blabel_padding:-blabel_padding, 1] == beat.value)[0] + blabel_padding

                    if enable_filter:  # Filter indices
                        fn = functools.partial(beat_filter_func, blabels=blabels, beat=beat)
                        beat_idxs = filter(fn, beat_idxs)
                    # END IF
                    pt_beat_map[beat_class] += [(seg_idx, blabels[i, 0]) for i in beat_idxs]
                # END FOR
            # END FOR
            pt_beat_map = [np.array(b) for b in pt_beat_map]
            self._pts_beat_map[patient_id] = pt_beat_map
        # END WITH

    def patient_data_generator(
        self,
        patient_id: int,
        samples_per_patient: list[int],
    ):
        """Generate data for given patient id"""
        input_size = int(np.ceil((self.ds.sampling_rate / self.sampling_rate) * self.frame_size))

        with self.ds.patient_data(patient_id) as segments:
            # This maps segment index to segment key
            seg_map: list[str] = list(segments.keys())
            if patient_id not in self._pts_beat_map:
                self._create_beat_map(patient_id)
            pt_beat_map = self._pts_beat_map[patient_id]

            # Randomly select N samples of each target beat
            pt_segs_beat_idxs: list[tuple[int, int, int]] = []
            for tgt_beat_idx, tgt_beats in enumerate(pt_beat_map):
                tgt_count = min(samples_per_patient[tgt_beat_idx], len(tgt_beats))
                tgt_idxs = np.random.choice(np.arange(len(tgt_beats)), size=tgt_count, replace=False)
                pt_segs_beat_idxs += [(tgt_beats[i][0], tgt_beats[i][1], tgt_beat_idx) for i in tgt_idxs]
            # END FOR

            # Shuffle all
            random.shuffle(pt_segs_beat_idxs)

            # Grab selected samples for patient
            samples = []
            for seg_idx, beat_idx, beat in pt_segs_beat_idxs:
                frame_start = max(0, beat_idx - int(random.uniform(0.4722, 0.5278) * input_size))
                frame_end = frame_start + input_size
                data = segments[seg_map[seg_idx]]["data"]
                x = np.nan_to_num(data[frame_start:frame_end]).astype(np.float32)
                if self.ds.sampling_rate != self.sampling_rate:
                    x = pk.signal.resample_signal(x, self.ds.sampling_rate, self.sampling_rate, axis=0)
                    x = x[: self.frame_size]  # truncate to frame size
                y = beat
                samples.append((x, y))
            # END FOR
        # END WITH

        # Yield samples
        for x, y in samples:
            yield x, y
        # END FOR

    def data_generator(
        self,
        patient_ids: list[int],
        samples_per_patient: int | list[int],
        shuffle: bool = False,
    ) -> Generator[tuple[npt.NDArray, npt.NDArray], None, None]:
        """Generate data for given patient ids"""
        # Target labels and mapping
        tgt_labels = sorted(set(self.label_map.values()))
        num_classes = len(tgt_labels)

        # If samples_per_patient is a list, then it must be the same length as nclasses
        if isinstance(samples_per_patient, Iterable):
            samples_per_tgt = samples_per_patient
        else:
            num_per_tgt = int(max(1, samples_per_patient / num_classes))
            samples_per_tgt = num_per_tgt * [num_classes]

        pt_ids = copy.deepcopy(patient_ids)
        for pt_id in helia.utils.uniform_id_generator(pt_ids, repeat=True, shuffle=shuffle):
            for x, y in self.patient_data_generator(pt_id, samples_per_tgt):
                yield x, y
            # END FOR
        # END FOR
