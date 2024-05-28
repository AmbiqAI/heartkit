from typing import Generator, Iterable

import numpy as np
import numpy.typing as npt
import physiokit as pk

from ....datasets import BidmcDataset, PatientGenerator


def bidmc_data_generator(
    patient_generator: PatientGenerator,
    ds: BidmcDataset,
    frame_size: int,
    samples_per_patient: int | list[int] = 1,
    target_rate: int | None = None,
) -> Generator[tuple[npt.NDArray, npt.NDArray], None, None]:
    """Generate frames using patient generator.

    Args:
        patient_generator (PatientGenerator): Patient Generator
        ds: BidmcDataset
        frame_size (int): Frame size
        samples_per_patient (int | list[int], optional): # samples per patient. Defaults to 1.
        target_rate (int|None, optional): Target rate. Defaults to None.

    Returns:
        Generator[tuple[npt.NDArray, npt.NDArray], None, None]: Sample generator

    """
    if isinstance(samples_per_patient, Iterable):
        samples_per_patient = samples_per_patient[0]

    for pt in patient_generator:
        with ds.patient_data(pt) as h5:
            ecg = h5["data"][0, :]
            ppg = h5["data"][1, :]
        # END WITH

        # Use translation map to determine source and target signals
        x = ppg
        y = ecg

        # Resample signals if necessary
        if ds.sampling_rate != target_rate:
            x = pk.signal.resample_signal(x, ds.sampling_rate, target_rate, axis=0)
            y = pk.signal.resample_signal(y, ds.sampling_rate, target_rate, axis=0)
        # END IF

        # Generate samples
        for _ in range(samples_per_patient):
            start = np.random.randint(0, x.size - frame_size)
            xx = x[start : start + frame_size]
            xx = np.nan_to_num(xx).astype(np.float32)
            yy = y[start : start + frame_size]
            yy = np.nan_to_num(yy).astype(np.float32)
            yield xx, yy
        # END FOR
    # END FOR
