import random
from typing import Generator

import numpy as np
import numpy.typing as npt
import physiokit as pk

from ....datasets.defines import PatientGenerator
from ....datasets.ludb import (
    FID_LOC_IDX,
    SEG_BEG_IDX,
    SEG_END_IDX,
    SEG_LBL_IDX,
    SEG_LEAD_IDX,
    LudbDataset,
    LudbSegmentation,
)
from ..defines import HKSegment

LudbSegmentationMap = {
    LudbSegmentation.normal: HKSegment.normal,
    LudbSegmentation.pwave: HKSegment.pwave,
    LudbSegmentation.qrs: HKSegment.qrs,
    LudbSegmentation.twave: HKSegment.twave,
}


def ludb_label_map(
    label_map: dict[int, int] | None = None,
) -> dict[int, int]:
    """Get label map

    Args:
        label_map (dict[int, int]|None): Label map

    Returns:
        dict[int, int]: Label map
    """
    return {k: label_map.get(v, -1) for (k, v) in LudbSegmentationMap.items()}


def ludb_data_generator(
    patient_generator: PatientGenerator,
    ds: LudbDataset,
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

    # Convert global labels -> ds labels -> class labels (-1 indicates not in class map)
    tgt_map = ludb_label_map(label_map)

    for pt in patient_generator:
        with ds.patient_data(pt) as h5:
            data = h5["data"][:]
            segs = h5["segmentations"][:]
            fids = h5["fiducials"][:]
        # END WITH

        if ds.sampling_rate != target_rate:
            ratio = target_rate / ds.sampling_rate
            data = pk.signal.resample_signal(data, ds.sampling_rate, target_rate, axis=0)
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
            lead = random.choice(ds.leads)
            # Randomly select frame within the segment
            frame_start = np.random.randint(start_offset, data.shape[0] - frame_size - stop_offset)
            frame_end = frame_start + frame_size
            x = data[frame_start:frame_end, lead].astype(np.float32)
            y = labels[frame_start:frame_end, lead].astype(np.int32)
            y = np.vectorize(tgt_map.get, otypes=[int])(y)
            yield x, y
        # END FOR
    # END FOR
