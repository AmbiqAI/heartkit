import random
from typing import Generator, Iterable

import numpy as np
import numpy.typing as npt
import physiokit as pk
import helia_edge as helia

from ....datasets import EcgSyntheticDataset, HKDataloader
from ..defines import HKSegment

EcgSyntheticSegmentationMap = {
    pk.ecg.EcgSegment.tp_overlap: HKSegment.pwave,
    pk.ecg.EcgSegment.p_wave: HKSegment.pwave,
    pk.ecg.EcgSegment.qrs_complex: HKSegment.qrs,
    pk.ecg.EcgSegment.t_wave: HKSegment.twave,
    pk.ecg.EcgSegment.background: HKSegment.normal,
    pk.ecg.EcgSegment.u_wave: HKSegment.uwave,
    pk.ecg.EcgSegment.pr_segment: HKSegment.normal,
    pk.ecg.EcgSegment.st_segment: HKSegment.normal,
    pk.ecg.EcgSegment.tp_segment: HKSegment.normal,
}


class EcgSyntheticDataloader(HKDataloader):
    def __init__(self, ds: EcgSyntheticDataset, **kwargs):
        """Dataloader for ECG synthetic dataset"""
        super().__init__(ds=ds, **kwargs)
        # Update label map
        if self.label_map:
            self.label_map = {
                k: self.label_map[v] for (k, v) in EcgSyntheticSegmentationMap.items() if v in self.label_map
            }
        # END DEF

    def patient_data_generator(
        self,
        patient_id: int,
        samples_per_patient: int,
    ):
        input_size = int(np.ceil((self.ds.sampling_rate / self.sampling_rate) * self.frame_size))

        start_offset = 0

        with self.ds.patient_data(patient_id) as h5:
            data = h5["data"][:]
            segs = h5["segmentations"][:]
        # END WITH

        for _ in range(samples_per_patient):
            lead = random.choice(self.ds.leads)
            start = np.random.randint(start_offset, data.shape[1] - input_size)
            x = data[lead, start : start + input_size].squeeze()
            x = np.nan_to_num(x).astype(np.float32)
            x = self.ds.add_noise(x)
            y = segs[lead, start : start + input_size].squeeze()
            y = np.vectorize(lambda v: self.label_map.get(v, 0), otypes=[int])(y)
            y = y.astype(np.int32)

            if self.ds.sampling_rate != self.sampling_rate:
                ratio = self.sampling_rate / self.ds.sampling_rate
                x = pk.signal.resample_signal(x, self.ds.sampling_rate, self.sampling_rate, axis=0)
                x = x[: self.frame_size]  # Ensure frame size
                y_tgt = np.zeros(x.shape, dtype=np.int32)
                start_idxs = np.hstack((0, np.nonzero(np.diff(y))[0]))
                end_idxs = np.hstack((start_idxs[1:], y.size))
                for s, e in zip(start_idxs, end_idxs):
                    y_tgt[int(s * ratio) : int(e * ratio)] = y[s]
                # END FOR
                y = y_tgt
            # END IF
            x = x.reshape(-1, 1)
            yield x, y
        # END FOR

    def data_generator(
        self,
        patient_ids: list[int],
        samples_per_patient: int | list[int],
        shuffle: bool = False,
    ) -> Generator[tuple[npt.NDArray, npt.NDArray], None, None]:
        if isinstance(samples_per_patient, Iterable):
            samples_per_patient = samples_per_patient[0]

        for pt_id in helia.utils.uniform_id_generator(patient_ids, shuffle=shuffle):
            for x, y in self.patient_data_generator(pt_id, samples_per_patient):
                yield x, y
            # END FOR
        # END FOR
