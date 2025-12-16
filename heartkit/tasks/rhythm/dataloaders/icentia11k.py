import random
from typing import Generator, Iterable

import numpy as np
import numpy.typing as npt
import physiokit as pk
import helia_edge as helia

from ....datasets import HKDataloader, IcentiaDataset, IcentiaRhythm
from ..defines import HKRhythm

IcentiaRhythmMap = {
    IcentiaRhythm.noise: HKRhythm.noise,
    IcentiaRhythm.normal: HKRhythm.sr,
    IcentiaRhythm.afib: HKRhythm.afib,
    IcentiaRhythm.aflut: HKRhythm.aflut,
    IcentiaRhythm.end: HKRhythm.noise,
}


class Icentia11kDataloader(HKDataloader):
    def __init__(self, ds: IcentiaDataset, **kwargs):
        """Dataloader for icentia11k dataset"""
        super().__init__(ds=ds, **kwargs)
        # Update label map
        if self.label_map:
            self.label_map = {k: self.label_map[v] for (k, v) in IcentiaRhythmMap.items() if v in self.label_map}
        # END DEF
        self.label_type = "rhythm"
        # PT: [label_idx, segment, start, end]
        self._pts_rhythm_map: dict[int, list[npt.NDArray]] = {}

    def _create_patient_rhythm_map(self, patient_id: int):
        # Target labels and mapping
        tgt_labels = sorted(set((self.label_map.values())))
        label_key = self.ds.label_key(self.label_type)

        input_size = int(np.ceil((self.ds.sampling_rate / self.sampling_rate) * self.frame_size))

        with self.ds.patient_data(patient_id=patient_id) as segments:
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
                xl = np.vectorize(self.label_map.get, otypes=[int])(xl)

                # Capture segment, start, and end for each target label
                for tgt_idx, tgt_class in enumerate(tgt_labels):
                    idxs = np.where((xe - xs >= input_size) & (xl == tgt_class))
                    seg_vals = np.vstack((seg_idx * np.ones_like(idxs), xs[idxs], xe[idxs])).T
                    pt_tgt_seg_map[tgt_idx] += seg_vals.tolist()
                # END FOR
            # END FOR

            pt_tgt_seg_map = [np.array(b) for b in pt_tgt_seg_map]
            self._pts_rhythm_map[patient_id] = pt_tgt_seg_map

    def patient_data_generator(
        self,
        patient_id: int,
        samples_per_patient: list[int],
    ):
        # Target labels and mapping
        tgt_labels = sorted(set(self.label_map.values()))

        input_size = int(np.ceil((self.ds.sampling_rate / self.sampling_rate) * self.frame_size))

        # Group patient rhythms by type (segment, start, stop, delta)

        with self.ds.patient_data(patient_id=patient_id) as segments:
            # This maps segment index to segment key
            seg_map: list[str] = list(segments.keys())
            if patient_id not in self._pts_rhythm_map:
                self._create_patient_rhythm_map(patient_id)
            pt_tgt_seg_map = self._pts_rhythm_map[patient_id]

            # Grab target segments
            seg_samples: list[tuple[int, int, int, int]] = []
            for tgt_idx, tgt_class in enumerate(tgt_labels):
                tgt_segments = pt_tgt_seg_map[tgt_idx]
                if not tgt_segments.shape[0]:
                    continue
                tgt_seg_indices: list[int] = random.choices(
                    np.arange(tgt_segments.shape[0]),
                    weights=tgt_segments[:, 2] - tgt_segments[:, 1],
                    k=samples_per_patient[tgt_idx],
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

            # Grab selected samples for patient
            samples = []
            for seg_idx, frame_start, frame_end, label in seg_samples:
                x: npt.NDArray = segments[seg_map[seg_idx]]["data"][frame_start:frame_end].astype(np.float32)
                if self.ds.sampling_rate != self.sampling_rate:
                    x = pk.signal.resample_signal(x, self.ds.sampling_rate, self.sampling_rate, axis=0)
                    x = x[: self.frame_size]  # truncate to frame size
                x = np.reshape(x, (self.frame_size, 1))
                samples.append((x, label))
            # END FOR
        # END WITH

        for x, y in samples:
            yield x, y
        # END FOR

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

        self._pts_beat_map = {}
        for pt_id in helia.utils.uniform_id_generator(patient_ids, shuffle=shuffle):
            for x, y in self.patient_data_generator(pt_id, samples_per_tgt):
                yield x, y
            # END FOR
        # END FOR
