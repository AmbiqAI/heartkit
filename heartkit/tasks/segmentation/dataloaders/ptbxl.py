import random
from typing import Generator, Iterable

import numpy as np
import numpy.typing as npt
import physiokit as pk
import helia_edge as helia

from ....datasets import HKDataloader, PtbxlDataset
from ..defines import HKSegment


class PtbxlDataloader(HKDataloader):
    def __init__(self, ds: PtbxlDataset, **kwargs):
        """Dataloader for ptbxl dataset"""
        super().__init__(ds=ds, **kwargs)

    def patient_data_generator(
        self,
        patient_id: int,
        samples_per_patient: int,
    ):
        input_size = int(np.ceil((self.ds.sampling_rate / self.sampling_rate) * self.frame_size))

        with self.ds.patient_data(patient_id) as h5:
            data = h5["data"][:]
            blabels = h5["blabels"][:]
        # END WITH

        # NOTE: Multiply by 5 to convert from 100 Hz to 500 Hz
        blabels[:, 0] = blabels[:, 0] * 5
        for _ in range(samples_per_patient):
            # Select random lead and start index
            lead = random.choice(self.ds.leads)
            frame_start = np.random.randint(0, data.shape[1] - input_size)
            frame_end = frame_start + input_size
            frame_blabels = blabels[(blabels[:, 0] >= frame_start) & (blabels[:, 0] < frame_end)]
            x = data[lead, frame_start:frame_end].copy()
            if self.ds.sampling_rate != self.sampling_rate:
                ds_ratio = self.sampling_rate / self.ds.sampling_rate
                x = pk.signal.resample_signal(x, self.ds.sampling_rate, self.sampling_rate, axis=0)
                x = x[: self.frame_size]  # Ensure frame size
            else:
                ds_ratio = 1

            # Create segment mask
            mask = np.zeros_like(x, dtype=np.int32)

            # # Check if pwave, twave, or uwave are in tgt_map- if so, add gradient filter to mask
            # non_qrs = [tgt_map.get(k, -1) for k in (HKSegment.pwave, HKSegment.twave, HKSegment.uwave)]
            # if any((v != -1 for v in non_qrs)):
            #     xc = pk.ecg.clean(x.copy(), sample_rate=target_rate, lowcut=0.5, highcut=40, order=3)
            #     grad = pk.signal.moving_gradient_filter(
            #         xc, sample_rate=target_rate, sig_window=0.1, avg_window=1.0, sig_prom_weight=0.15
            #     )
            #     mask[grad > 0] = -1
            # # END IF

            for i in range(frame_blabels.shape[0]):
                bidx = int((frame_blabels[i, 0] - frame_start) * ds_ratio)
                # btype = frame_blabels[i, 1]

                # Extract QRS segment
                qrs = pk.signal.moving_gradient_filter(
                    x, sample_rate=self.sampling_rate, sig_window=0.1, avg_window=1.0, sig_prom_weight=1.5
                )
                win_len = max(1, int(0.08 * self.sampling_rate))  # 80 ms
                b_left = max(0, bidx - win_len)
                b_right = min(x.shape[0], bidx + win_len)
                onset = np.where(np.flip(qrs[b_left:bidx]) < 0)[0]
                onset = onset[0] if onset.size else win_len
                offset = np.where(qrs[bidx + 1 : b_right] < 0)[0]
                offset = offset[0] if offset.size else win_len
                mask[bidx - onset : bidx + offset] = self.label_map.get(HKSegment.qrs.value, 0)
                # END IF
            # END FOR
            x = np.nan_to_num(x).astype(np.float32)
            x = x.reshape(-1, 1)
            y = mask.astype(np.int32)
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
