from typing import Generator

import numpy as np
import numpy.typing as npt
import physiokit as pk

from ....datasets.defines import PatientGenerator
from ....datasets.syntheticppg import SyntheticPpgDataset
from ..defines import HKSegment

SyntheticPpgSegmentationMap = {
    pk.ppg.PpgSegment.background: HKSegment.normal,
    pk.ppg.PpgSegment.systolic: HKSegment.systolic,
    pk.ppg.PpgSegment.diastolic: HKSegment.diastolic,
}


def syntheticppg_label_map(
    label_map: dict[int, int] | None = None,
) -> dict[int, int]:
    """Get label map

    Args:
        label_map (dict[int, int]|None): Label map

    Returns:
        dict[int, int]: Label map
    """
    return {k: label_map[v] for (k, v) in SyntheticPpgSegmentationMap.items() if v in label_map}


def syntheticppg_data_generator(
    patient_generator: PatientGenerator,
    ds: SyntheticPpgDataset,
    frame_size: int,
    samples_per_patient: int | list[int] = 1,
    target_rate: int | None = None,
    label_map: dict[int, int] | None = None,
) -> Generator[tuple[npt.NDArray, npt.NDArray], None, None]:
    """Generate frames w/ rhythm labels (e.g. afib) using patient generator.

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

    input_size = int(np.round((ds.sampling_rate / target_rate) * frame_size))

    tgt_map = syntheticppg_label_map(label_map)

    start_offset = 0

    for pt in patient_generator:
        with ds.patient_data(pt) as h5:
            data = h5["data"][:]
            segs = h5["segmentations"][:]
        # END WITH

        for _ in range(samples_per_patient):
            start = np.random.randint(start_offset, data.shape[0] - input_size)
            x = data[start : start + input_size].squeeze()
            x = np.nan_to_num(x).astype(np.float32)
            x = ds.add_noise(x)
            y = segs[start : start + input_size].squeeze()
            y = y.astype(np.int32)
            y = np.vectorize(lambda v: tgt_map.get(v, 0), otypes=[int])(y)

            if ds.sampling_rate != target_rate:
                ratio = target_rate / ds.sampling_rate
                x = pk.signal.resample_signal(x, ds.sampling_rate, target_rate, axis=0)
                y_tgt = np.zeros(x.shape, dtype=np.int32)
                start_idxs = np.hstack((0, np.nonzero(np.diff(y))[0]))
                end_idxs = np.hstack((start_idxs[1:], y.size))
                for s, e in zip(start_idxs, end_idxs):
                    y_tgt[int(s * ratio) : int(e * ratio)] = y[s]
                # END FOR
                y = y_tgt
            # END IF
            yield x, y
        # END FOR
    # END FOR
