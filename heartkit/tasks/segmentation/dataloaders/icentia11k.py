from typing import Generator, Iterable

import numpy as np
import numpy.typing as npt
import physiokit as pk

from ....datasets.defines import PatientGenerator
from ....datasets.icentia11k import IcentiaBeat, IcentiaDataset
from ..defines import HKSegment


def icentia11k_label_map(
    label_map: dict[int, int] | None = None,
) -> dict[int, int]:
    """Get label map

    Args:
        label_map (dict[int, int]|None): Label map

    Returns:
        dict[int, int]: Label map
    """
    return label_map


def icentia11k_data_generator(
    patient_generator: PatientGenerator,
    ds: IcentiaDataset,
    frame_size: int,
    samples_per_patient: int | list[int] = 1,
    target_rate: int | None = None,
    label_map: dict[int, int] | None = None,
) -> Generator[tuple[npt.NDArray, npt.NDArray], None, None]:
    """Generate frames w/ segmentation labels (e.g. qrs) using patient generator.

    Args:
        patient_generator (PatientGenerator): Patient Generator
        ds: IcentiaDataset
        frame_size (int): Frame size
        samples_per_patient (int | list[int], optional): # samples per patient. Defaults to 1.
        target_rate (int|None, optional): Target rate. Defaults to None.
        label_map (dict[int, int] | None, optional): Label map. Defaults to None.

    Returns:
        Generator[tuple[npt.NDArray, npt.NDArray], None, None]: Sample generator
    """

    if target_rate is None:
        target_rate = ds.sampling_rate
    # END IF

    if isinstance(samples_per_patient, Iterable):
        samples_per_patient = samples_per_patient[0]

    tgt_map = label_map  # We generate the labels in the generator

    input_size = int(np.round((ds.sampling_rate / target_rate) * frame_size))

    # For each patient
    for pt in patient_generator:
        with ds.patient_data(pt) as segments:
            for _ in range(samples_per_patient):
                # Randomly pick a segment
                seg_key = np.random.choice(list(segments.keys()))
                # Randomly pick a frame
                frame_start = np.random.randint(segments[seg_key]["data"].shape[0] - input_size)
                frame_end = frame_start + input_size
                # Get data and labels
                data = segments[seg_key]["data"][frame_start:frame_end].squeeze()

                if ds.sampling_rate != target_rate:
                    ds_ratio = target_rate / ds.sampling_rate
                    data = pk.signal.resample_signal(data, ds.sampling_rate, target_rate, axis=0)
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
                        # noise_lbl = self.class_map.get(HeartSegment.noise.value, -1)
                        # # Skip if not in class map
                        # if noise_lbl == -1
                        #     continue
                        # # Mark region as noise
                        # win_len = max(1, int(0.2 * self.target_rate))  # 200 ms
                        # b_left = max(0, bidx - win_len)
                        # b_right = min(data.shape[0], bidx + win_len)
                        # mask[b_left:b_right] = noise_lbl

                    # Normal, PAC, PVC beat
                    else:
                        qrs_width = int(0.08 * target_rate)  # 80 ms
                        # Extract QRS segment
                        qrs = pk.signal.moving_gradient_filter(
                            data,
                            sample_rate=target_rate,
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
                        mask[qrs_onset:qrs_offset] = tgt_map.get(HKSegment.qrs.value, 0)
                    # END IF
                # END FOR
                x = np.nan_to_num(data).astype(np.float32)
                y = mask.astype(np.int32)
                yield x, y
            # END FOR
        # END WITH
    # END FOR
