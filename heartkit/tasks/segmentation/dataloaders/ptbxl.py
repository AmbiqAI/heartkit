import random
from typing import Generator, Iterable

import numpy as np
import numpy.typing as npt
import physiokit as pk

from ....datasets.defines import PatientGenerator
from ....datasets.ptbxl import PtbxlDataset
from ..defines import HKSegment


def ptbxl_label_map(
    label_map: dict[int, int] | None = None,
) -> dict[int, int]:
    """Get label map

    Args:
        label_map (dict[int, int]|None): Label map

    Returns:
        dict[int, int]: Label map
    """

    # We generate the labels in the generator
    return label_map


def ptbxl_data_generator(
    patient_generator: PatientGenerator,
    ds: PtbxlDataset,
    frame_size: int,
    samples_per_patient: int | list[int] = 1,
    target_rate: int | None = None,
    label_map: dict[int, int] | None = None,
) -> Generator[tuple[npt.NDArray, npt.NDArray], None, None]:
    """Generate frames w/ segmentation labels (e.g. qrs) using patient generator.

    Args:
        patient_generator (PatientGenerator): Patient Generator
        ds: PtbxlDataset
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

    tgt_map = ptbxl_label_map(label_map=label_map)

    input_size = int(np.round((ds.sampling_rate / target_rate) * frame_size))

    # For each patient
    for pt in patient_generator:
        with ds.patient_data(pt) as h5:
            data = h5["data"][:]
            blabels = h5["blabels"][:]
        # END WITH

        # NOTE: Multiply by 5 to convert from 100 Hz to 500 Hz
        blabels[:, 0] = blabels[:, 0] * 5
        for _ in range(samples_per_patient):
            # Select random lead and start index
            lead = random.choice(ds.leads)
            frame_start = np.random.randint(0, data.shape[1] - input_size)
            frame_end = frame_start + input_size
            frame_blabels = blabels[(blabels[:, 0] >= frame_start) & (blabels[:, 0] < frame_end)]
            x = data[lead, frame_start:frame_end].copy()
            if ds.sampling_rate != target_rate:
                ds_ratio = target_rate / ds.sampling_rate
                x = pk.signal.resample_signal(x, ds.sampling_rate, target_rate, axis=0)
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
                    x, sample_rate=target_rate, sig_window=0.1, avg_window=1.0, sig_prom_weight=1.5
                )
                win_len = max(1, int(0.08 * target_rate))  # 80 ms
                b_left = max(0, bidx - win_len)
                b_right = min(x.shape[0], bidx + win_len)
                onset = np.where(np.flip(qrs[b_left:bidx]) < 0)[0]
                onset = onset[0] if onset.size else win_len
                offset = np.where(qrs[bidx + 1 : b_right] < 0)[0]
                offset = offset[0] if offset.size else win_len
                mask[bidx - onset : bidx + offset] = tgt_map.get(HKSegment.qrs.value, 0)
                # END IF
            # END FOR
            x = np.nan_to_num(x).astype(np.float32)
            y = mask.astype(np.int32)
            yield x, y
        # END FOR
    # END FOR
