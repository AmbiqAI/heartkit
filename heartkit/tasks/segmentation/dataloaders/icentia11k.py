from typing import Generator, Iterable

import numpy as np
import numpy.typing as npt
import physiokit as pk
import helia_edge as helia

from ....datasets import IcentiaBeat, IcentiaDataset, HKDataloader
from ..defines import HKSegment


class Icentia11kDataloader(HKDataloader):
    def __init__(self, ds: IcentiaDataset, **kwargs):
        """Dataloader for icentia11k dataset"""
        super().__init__(ds=ds, **kwargs)

    def patient_data_generator(
        self,
        patient_id: int,
        samples_per_patient: int,
    ):
        input_size = int(np.ceil((self.ds.sampling_rate / self.sampling_rate) * self.frame_size))

        with self.ds.patient_data(patient_id) as segments:
            for _ in range(samples_per_patient):
                # Randomly pick a segment
                seg_key = np.random.choice(list(segments.keys()))
                # Randomly pick a frame
                frame_start = np.random.randint(segments[seg_key]["data"].shape[0] - input_size)
                frame_end = frame_start + input_size
                # Get data and labels
                data = segments[seg_key]["data"][frame_start:frame_end].squeeze()

                if self.ds.sampling_rate != self.sampling_rate:
                    ds_ratio = self.sampling_rate / self.ds.sampling_rate
                    data = pk.signal.resample_signal(data, self.ds.sampling_rate, self.sampling_rate, axis=0)
                    data = data[: self.frame_size]  # Ensure frame size
                else:
                    ds_ratio = 1

                blabels = segments[seg_key]["blabels"]
                blabels = blabels[(blabels[:, 0] >= frame_start) & (blabels[:, 0] < frame_end)]
                # Create segment mask
                mask = np.zeros_like(data, dtype=np.int32)

                # # Check if pwave, twave, or uwave are in class_map- if so, add gradient filter to mask
                # non_qrs = [self.class_map.get(k, -1) for k in (HKSegment.pwave, HKSegment.twave, HKSegment.uwave)]
                # if any((v != -1 for v in non_qrs)):
                #     xc = pk.ecg.clean(data.copy(), sample_rate=self.target_rate, lowcut=0.5, highcut=40, order=3)
                #     grad = pk.signal.moving_gradient_filter(
                #         xc, sample_rate=self.target_rate, sig_window=0.1, avg_window=1.0, sig_prom_weight=0.15
                #     )
                #     mask[grad > 0] = -1
                # # END IF

                for i in range(blabels.shape[0]):
                    bidx = int((blabels[i, 0] - frame_start) * ds_ratio)
                    btype = blabels[i, 1]
                    # Unclassifiable beat (treat as noise?)
                    if btype == IcentiaBeat.undefined:
                        pass
                    # Normal, PAC, PVC beat
                    else:
                        qrs_width = int(0.08 * self.sampling_rate)  # 80 ms
                        # Extract QRS segment
                        qrs = pk.signal.moving_gradient_filter(
                            data,
                            sample_rate=self.sampling_rate,
                            sig_window=0.1,
                            avg_window=1.0,
                            sig_prom_weight=1.5,
                        )
                        win_len = max(1, qrs_width)
                        b_left = max(0, bidx - win_len)
                        b_right = min(data.shape[0], bidx + win_len)
                        onset = np.where(np.flip(qrs[b_left:bidx]) < 0)[0]
                        onset = onset[0] if onset.size else win_len
                        offset = np.where(qrs[bidx + 1 : b_right] < 0)[0]
                        offset = offset[0] if offset.size else win_len
                        qrs_onset = bidx - onset
                        qrs_offset = bidx + offset
                        mask[qrs_onset:qrs_offset] = self.label_map.get(HKSegment.qrs.value, 0)
                    # END IF
                # END FOR
                x = np.nan_to_num(data).astype(np.float32)
                x = x.reshape(-1, 1)
                y = mask.astype(np.int32)
                yield x, y
            # END FOR
        # END WITH

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
