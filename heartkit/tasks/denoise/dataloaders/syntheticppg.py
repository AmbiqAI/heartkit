from typing import Generator, Iterable

import numpy.typing as npt

from ....datasets import PatientGenerator, SyntheticPpgDataset


def synthetic_ppg_data_generator(
    patient_generator: PatientGenerator,
    ds: SyntheticPpgDataset,
    frame_size: int,
    samples_per_patient: int | list[int] = 1,
    target_rate: int | None = None,
) -> Generator[tuple[npt.NDArray, npt.NDArray], None, None]:
    """Generate frames using patient generator.

    Args:
        patient_generator (PatientGenerator): Patient Generator
        ds: SyntheticPpgDataset
        frame_size (int): Frame size
        samples_per_patient (int | list[int], optional): # samples per patient. Defaults to 1.
        target_rate (int|None, optional): Target rate. Defaults to None.

    Returns:
        Generator[tuple[npt.NDArray, npt.NDArray], None, None]: Sample generator

    """
    if isinstance(samples_per_patient, Iterable):
        samples_per_patient = samples_per_patient[0]

    gen = ds.signal_generator(
        patient_generator=patient_generator,
        frame_size=frame_size,
        samples_per_patient=samples_per_patient,
        target_rate=target_rate,
    )
    for x in gen:
        y = x.copy()
        yield x, y
    # END FOR
