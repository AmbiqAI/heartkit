from typing import Generator, Iterable

import numpy as np
import numpy.typing as npt
import physiokit as pk
import helia_edge as helia

from ....datasets import BidmcDataset, HKDataloader
from ..defines import HKTranslate


BidmcTranslateMap = {0: HKTranslate.ecg, 1: HKTranslate.ppg}


class BidmcDataloader(HKDataloader):
    """Dataloader for the BIDMC dataset"""

    def __init__(self, ds: BidmcDataset, **kwargs):
        super().__init__(ds=ds, **kwargs)
        if self.label_map is None:
            self.label_map = {HKTranslate.ppg: HKTranslate.ecg}
        if len(self.label_map) != 1:
            raise ValueError("Only one source and target signal is supported")
        self.label_map = {k: self.label_map[v] for (k, v) in BidmcTranslateMap.items() if k in self.label_map}

    def patient_data_generator(
        self,
        patient_id: int,
        samples_per_patient: int,
    ):
        # Use class_map to determine source and target signals
        src, tgt = list(self.label_map.keys())[0], list(self.label_map.values())[0]

        with self.ds.patient_data(patient_id) as h5:
            x = h5["data"][src, :]
            y = h5["data"][tgt, :]
        # END WITH

        # Resample signals if necessary
        if self.ds.sampling_rate != self.sampling_rate:
            x = pk.signal.resample_signal(x, self.ds.sampling_rate, self.sampling_rate, axis=0)
            y = pk.signal.resample_signal(y, self.ds.sampling_rate, self.sampling_rate, axis=0)
        # END IF

        # Generate samples
        for _ in range(samples_per_patient):
            start = np.random.randint(0, x.size - self.frame_size)
            xx = x[start : start + self.frame_size]
            xx = np.nan_to_num(xx).astype(np.float32)
            yy = y[start : start + self.frame_size]
            yy = np.nan_to_num(yy).astype(np.float32)
            xx = xx.reshape(-1, 1)
            yy = yy.reshape(-1, 1)
            yield xx, yy
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
