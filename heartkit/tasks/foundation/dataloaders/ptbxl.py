import random
from typing import Generator

import numpy as np
import numpy.typing as npt
import physiokit as pk

from ....datasets import PatientGenerator, PtbxlDataset


def ptbxl_data_generator(
    patient_generator: PatientGenerator,
    ds: PtbxlDataset,
    frame_size: int,
    samples_per_patient: int | list[int] = 1,
    target_rate: int | None = None,
) -> Generator[tuple[npt.NDArray, npt.NDArray], None, None]:
    """Generate frames using patient generator.

    Args:
        patient_generator (PatientGenerator): Patient Generator
        ds: PtbxlDataset
        frame_size (int): Frame size
        samples_per_patient (int | list[int], optional): # samples per patient. Defaults to 1.
        target_rate (int|None, optional): Target rate. Defaults to None.

    Returns:
        Generator[tuple[npt.NDArray, npt.NDArray], None, None]: Sample generator

    """
    input_size = int(np.round((ds.sampling_rate / target_rate) * frame_size))
    data_cache = {}
    for pt in patient_generator:
        if pt not in data_cache:
            with ds.patient_data(pt) as h5:
                data_cache[pt] = h5["data"][:]
        data = data_cache[pt]
        # with ds.patient_data(pt) as h5:
        #     data = h5["data"][:]

        for _ in range(samples_per_patient):
            leads = random.sample(ds.leads, k=2)
            lead_p1 = leads[0]
            lead_p2 = leads[1]
            start_p1 = np.random.randint(0, data.shape[1] - input_size)
            start_p2 = np.random.randint(0, data.shape[1] - input_size)
            # start_p2 = start_p1

            x1 = np.nan_to_num(data[lead_p1, start_p1 : start_p1 + input_size].squeeze()).astype(np.float32)
            x2 = np.nan_to_num(data[lead_p2, start_p2 : start_p2 + input_size].squeeze()).astype(np.float32)

            if ds.sampling_rate != target_rate:
                x1 = pk.signal.resample_signal(x1, ds.sampling_rate, target_rate, axis=0)
                x2 = pk.signal.resample_signal(x2, ds.sampling_rate, target_rate, axis=0)
            # END IF
            yield x1, x2
        # END FOR
    # END FOR
