import random
from typing import Generator, Iterable

import numpy as np
import numpy.typing as npt
import physiokit as pk
import helia_edge as helia

from ....datasets import HKDataloader, LudbDataset, LudbSegmentation
from ....datasets.ludb import FID_LOC_IDX, SEG_BEG_IDX, SEG_END_IDX, SEG_LBL_IDX, SEG_LEAD_IDX
from ..defines import HKSegment

LudbSegmentationMap = {
    LudbSegmentation.normal: HKSegment.normal,
    LudbSegmentation.pwave: HKSegment.pwave,
    LudbSegmentation.qrs: HKSegment.qrs,
    LudbSegmentation.twave: HKSegment.twave,
}


class LudbDataloader(HKDataloader):
    def __init__(self, ds: LudbDataset, **kwargs):
        """Dataloader for ludb dataset"""
        super().__init__(ds=ds, **kwargs)
        if self.label_map:
            self.label_map = {k: self.label_map[v] for (k, v) in LudbSegmentationMap.items() if v in self.label_map}

    def patient_data_generator(
        self,
        patient_id: int,
        samples_per_patient: int,
    ):
        with self.ds.patient_data(patient_id) as h5:
            data = h5["data"][:].copy()
            segs = h5["segmentations"][:].copy()
            fids = h5["fiducials"][:].copy()
        # END WITH

        if self.ds.sampling_rate != self.sampling_rate:
            ratio = self.sampling_rate / self.ds.sampling_rate
            data = pk.signal.resample_signal(data, self.ds.sampling_rate, self.sampling_rate, axis=0)
            segs[:, (SEG_BEG_IDX, SEG_END_IDX)] = segs[:, (SEG_BEG_IDX, SEG_END_IDX)] * ratio
            fids[:, FID_LOC_IDX] = fids[:, FID_LOC_IDX] * ratio
        # END IF

        # Create segmentation mask
        labels = np.zeros_like(data)
        for seg_idx in range(segs.shape[0]):
            seg = segs[seg_idx]
            labels[seg[SEG_BEG_IDX] : seg[SEG_END_IDX], seg[SEG_LEAD_IDX]] = seg[SEG_LBL_IDX]
        # END FOR

        start_offset = max(0, segs[0][SEG_BEG_IDX] - 100)
        stop_offset = max(0, data.shape[0] - segs[-1][SEG_END_IDX] + 100)
        for _ in range(samples_per_patient):
            # Randomly pick an ECG lead
            lead = random.choice(self.ds.leads)
            # Randomly select frame within the segment
            frame_start = np.random.randint(start_offset, data.shape[0] - self.frame_size - stop_offset)
            frame_end = frame_start + self.frame_size
            x = data[frame_start:frame_end, lead]
            x = np.nan_to_num(x, neginf=0, posinf=0).astype(np.float32)
            x = np.reshape(x, (-1, 1))
            y = labels[frame_start:frame_end, lead]
            y = np.vectorize(lambda v: self.label_map.get(v, 0), otypes=[int])(y)
            y = y.astype(np.int32)
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
