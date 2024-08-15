import copy
import random
from typing import Generator, Iterable

import numpy as np
import numpy.typing as npt

from ....datasets import HKDataloader, IcentiaMiniDataset, IcentiaMiniBeat
from ..defines import HKBeat

IcentiaBeatMap = {
    IcentiaMiniBeat.normal: HKBeat.normal,
    IcentiaMiniBeat.pac: HKBeat.pac,
    IcentiaMiniBeat.aberrated: HKBeat.pac,
    IcentiaMiniBeat.pvc: HKBeat.pvc,
}


class IcentiaMiniDataloader(HKDataloader):
    def __init__(self, ds: IcentiaMiniDataset, **kwargs):
        """IcentiaMini Dataloader for training beat tasks"""
        super().__init__(ds=ds, **kwargs)
        # Update label map
        if self.label_map:
            self.label_map = {k: self.label_map[v] for (k, v) in IcentiaBeatMap.items() if v in self.label_map}
        self.label_type = "beat"

    def data_generator(
        self,
        patient_ids: list[int],
        samples_per_patient: int | list[int],
        shuffle: bool = False,
    ) -> Generator[tuple[npt.NDArray, npt.NDArray], None, None]:
        """Generate data for given patient ids

        Args:
            patient_ids (list[int]): Patient IDs
            samples_per_patient (int | list[int]): Samples per patient
            shuffle (bool, optional): Shuffle data. Defaults to False.

        Yields:
            Generator[tuple[npt.NDArray, npt.NDArray], None, None]: Data generator
        """
        # Target labels and mapping
        tgt_labels = sorted(list(set((lbl for lbl in self.label_map.values() if lbl != -1))))
        label_key = self.ds.label_key(self.label_type)

        num_classes = len(tgt_labels)

        # If samples_per_patient is a list, then it must be the same length as nclasses
        if isinstance(samples_per_patient, Iterable):
            samples_per_tgt = samples_per_patient
        else:
            num_per_tgt = int(max(1, samples_per_patient / num_classes))
            samples_per_tgt = num_per_tgt * [num_classes]

        input_size = int(np.ceil((self.ds.sampling_rate / self.sampling_rate) * self.frame_size))
        print(f"Input size: {input_size} {samples_per_tgt}")

        pt_ids = copy.deepcopy(patient_ids)
        while True:
            for pt_id in pt_ids:
                with self.ds.patient_data(pt_id) as pt:
                    # data = pt["data"][:]  # has shape (N, M, 1)
                    # blabels is a mask with shape (N, M)
                    blabels = pt[label_key][:]

                    # Capture all beat locations
                    pt_beat_map = {}
                    for beat in IcentiaMiniBeat:
                        # Skip if not in class map
                        beat_class = self.label_map.get(beat, -1)
                        if beat_class < 0 or beat_class >= num_classes:
                            continue
                        # Get all beat type indices
                        rows, cols = np.where(blabels == beat.value)
                        # Zip rows and cols to form N, 2 array
                        pt_beat_map[beat_class] = np.array(list(zip(rows, cols)))
                    # END FOR
                # END WITH
                for samples in samples_per_patient:
                    for i in range(samples):
                        yield np.random.normal(size=(self.frame_size, 1)), np.random.randint(0, num_classes)
                # END FOR

            # END FOR
            if shuffle:
                random.shuffle(pt_ids)
