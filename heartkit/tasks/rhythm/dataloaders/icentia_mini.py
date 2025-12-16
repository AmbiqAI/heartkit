import random
from typing import Generator, Iterable

import numpy as np
import numpy.typing as npt
import physiokit as pk
import helia_edge as helia

from ....datasets import HKDataloader, IcentiaMiniDataset, IcentiaMiniRhythm
from ..defines import HKRhythm

IcentiaMiniRhythmMap = {
    IcentiaMiniRhythm.normal: HKRhythm.sr,
    IcentiaMiniRhythm.afib: HKRhythm.afib,
    IcentiaMiniRhythm.aflut: HKRhythm.aflut,
    IcentiaMiniRhythm.end: HKRhythm.noise,
}


class IcentiaMiniDataloader(HKDataloader):
    def __init__(self, ds: IcentiaMiniDataset, **kwargs):
        """Dataloader for icentia mini dataset"""
        super().__init__(ds=ds, **kwargs)
        # Update label map to map icentia mini label -> rhythm label -> user label
        if self.label_map:
            self.label_map = {k: self.label_map[v] for (k, v) in IcentiaMiniRhythmMap.items() if v in self.label_map}
        self.label_type = "rhythm"
        self._pts_rhythm_map: dict[int, dict[int, tuple[int, int, int]]] = {}

    def _create_patient_rhythm_map(self, patient_id: int):
        label_key = self.ds.label_key(self.label_type)
        tgt_labels = sorted(set(self.label_map.values()))
        # input_size = int(np.ceil((self.ds.sampling_rate / self.sampling_rate) * self.frame_size))

        pt_rhythm_map = {lbl: [] for lbl in tgt_labels}
        with self.ds.patient_data(patient_id) as pt:
            # rlabels is a mask with shape (N, M)
            rlabels = pt[label_key][:]

            # Capture all rhythm locations
            self.pts_rhythm_map: dict[int, tuple[int, int, int]] = {lbl: [] for lbl in tgt_labels}
            for r in range(rlabels.shape[0]):
                # Grab start and end locations by diffing the mask
                starts = np.concatenate(([0], np.where(np.abs(np.diff(rlabels[r, :])) >= 1)[0]))
                ends = np.concatenate((starts[1:], [rlabels.shape[1]]))
                lengths = ends - starts
                labels = rlabels[r, starts]
                # iterate through the zip of labels, starts, ends and append to the rhythm map
                for label, start, length in zip(labels, starts, lengths):
                    # Skip if label is not in the label map
                    if label not in self.label_map:
                        continue
                    # # Skip if the segment is too short
                    # if length < input_size:
                    #     continue
                    pt_rhythm_map[self.label_map[label]].append((r, start, length))
                # END FOR
            # END FOR
        # END WITH
        self._pts_rhythm_map[patient_id] = pt_rhythm_map

    def patient_data_generator(
        self,
        patient_id: int,
        samples_per_patient: list[int],
    ):
        tgt_labels = sorted(set(self.label_map.values()))
        input_size = int(np.ceil((self.ds.sampling_rate / self.sampling_rate) * self.frame_size))

        # Create rhythm map for all patients if needed
        if patient_id not in self._pts_rhythm_map:
            self._create_patient_rhythm_map(patient_id)

        with self.ds.patient_data(patient_id) as pt:
            data = pt["data"][:]  # has shape (N, M, 1)
            pt_rhythm_map = self._pts_rhythm_map[patient_id]
            for i, samples in enumerate(samples_per_patient):
                tgt_label = tgt_labels[i]
                locs = pt_rhythm_map.get(tgt_label, None)
                if not locs:
                    continue
                loc_indices = random.choices(range(len(locs)), k=samples)
                for loc_idx in loc_indices:
                    row, start, length = locs[loc_idx]
                    frame_start = max(0, random.randint(start, max(start, start + length - input_size) + 1))
                    frame_end = frame_start + input_size
                    x = data[row, frame_start:frame_end].astype(np.float32)
                    if self.ds.sampling_rate != self.sampling_rate:
                        x = pk.signal.resample_signal(x, self.ds.sampling_rate, self.sampling_rate, axis=0)
                        x = x[: self.frame_size]  # truncate to frame size
                    yield x, tgt_label
                # END FOR
            # END FOR
        # END WITH

    def data_generator(
        self,
        patient_ids: list[int],
        samples_per_patient: int | list[int],
        shuffle: bool = False,
    ) -> Generator[tuple[npt.NDArray, npt.NDArray], None, None]:
        # Target labels and mapping
        tgt_labels = sorted(set(self.label_map.values()))
        num_classes = len(tgt_labels)

        # If samples_per_patient is a list, then it must be the same length as nclasses
        if isinstance(samples_per_patient, Iterable):
            samples_per_tgt = samples_per_patient
        else:
            num_per_tgt = int(max(1, samples_per_patient / num_classes))
            samples_per_tgt = num_per_tgt * [num_classes]

        for pt_id in helia.utils.uniform_id_generator(patient_ids, shuffle=shuffle):
            for x, y in self.patient_data_generator(pt_id, samples_per_tgt):
                yield x, y
            # END FOR
        # END FOR
