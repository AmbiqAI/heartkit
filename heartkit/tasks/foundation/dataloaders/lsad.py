import random
from typing import Generator

import numpy as np
import numpy.typing as npt
import physiokit as pk
import helia_edge as helia

from ....datasets import HKDataloader, LsadDataset


class LsadDataloader(HKDataloader):
    def __init__(self, ds: LsadDataset, **kwargs):
        """Lsad Dataloader for training foundation tasks

        Args:
            ds (LsadDataset): LsadDataset
        """
        super().__init__(ds=ds, **kwargs)

    def patient_data_generator(
        self,
        patient_id: int,
        samples_per_patient: list[int],
    ):
        """Generate data for given patient id"""
        input_size = int(np.ceil((self.ds.sampling_rate / self.sampling_rate) * self.frame_size))

        with self.ds.patient_data(patient_id) as pt:
            data = pt["data"][:]

            for _ in range(samples_per_patient):
                leads = random.sample(self.ds.leads, k=2)
                lead_p1 = leads[0]
                lead_p2 = leads[1]
                start_p1 = np.random.randint(0, data.shape[1] - input_size)
                start_p2 = np.random.randint(0, data.shape[1] - input_size)
                # start_p2 = start_p1

                x1 = np.nan_to_num(data[lead_p1, start_p1 : start_p1 + input_size].squeeze()).astype(np.float32)
                x2 = np.nan_to_num(data[lead_p2, start_p2 : start_p2 + input_size].squeeze()).astype(np.float32)

                if self.ds.sampling_rate != self.sampling_rate:
                    x1 = pk.signal.resample_signal(x1, self.ds.sampling_rate, self.sampling_rate, axis=0)
                    x2 = pk.signal.resample_signal(x2, self.ds.sampling_rate, self.sampling_rate, axis=0)
                    x1 = x1[: self.frame_size]
                    x2 = x2[: self.frame_size]
                # END IF
                x1 = np.reshape(x1, (-1, 1))
                x2 = np.reshape(x2, (-1, 1))
                yield x1, x2
            # END FOR

    def data_generator(
        self,
        patient_ids: list[int],
        samples_per_patient: int | list[int],
        shuffle: bool = False,
    ) -> Generator[tuple[npt.NDArray, npt.NDArray], None, None]:
        for pt_id in helia.utils.uniform_id_generator(patient_ids, shuffle=shuffle):
            for x1, x2 in self.patient_data_generator(pt_id, samples_per_patient):
                yield x1, x2
            # END FOR
        # END FOR
