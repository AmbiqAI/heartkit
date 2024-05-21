from typing import Generator

import numpy.typing as npt

from ....datasets.defines import PatientGenerator
from ....datasets.lsad import LsadDataset, LsadScpCode
from ..defines import HKRhythm

LsadRhythmMap = {
    LsadScpCode.SR: HKRhythm.sr,
    LsadScpCode.SB: HKRhythm.sbrad,
    LsadScpCode.SBRAD: HKRhythm.sbrad,
    LsadScpCode.ST: HKRhythm.stach,
    LsadScpCode.AA: HKRhythm.sarrh,
    LsadScpCode.SA: HKRhythm.sarrh,
    LsadScpCode.AVNRT: HKRhythm.svt,
    LsadScpCode.AVNRT2: HKRhythm.svt,
    LsadScpCode.AVRT: HKRhythm.svt,
    LsadScpCode.SVT: HKRhythm.svt,
    LsadScpCode.WPW: HKRhythm.svt,
    LsadScpCode.AT: HKRhythm.svt,
    LsadScpCode.JT: HKRhythm.svt,
    LsadScpCode.PVT: HKRhythm.vtach,
    LsadScpCode.AFIB: HKRhythm.afib,
    LsadScpCode.AF: HKRhythm.aflut,
    LsadScpCode.VFIB: HKRhythm.vfib,
    LsadScpCode.VF: HKRhythm.vflut,
    LsadScpCode.ABI: HKRhythm.bigu,
    LsadScpCode.VB: HKRhythm.bigu,
    LsadScpCode.VET: HKRhythm.trigu,
}


def lsad_label_map(
    label_map: dict[int, int] | None = None,
) -> dict[int, int]:
    """Get label map

    Args:
        label_map (dict[int, int]|None): Label map

    Returns:
        dict[int, int]: Label map
    """
    return {k: label_map.get(v, -1) for (k, v) in LsadRhythmMap.items()}


def lsad_data_generator(
    patient_generator: PatientGenerator,
    ds: LsadDataset,
    frame_size: int,
    samples_per_patient: int | list[int] = 1,
    target_rate: int | None = None,
    label_map: dict[int, int] | None = None,
) -> Generator[tuple[npt.NDArray, int], None, None]:
    """Generate frames w/ rhythm labels (e.g. afib) using patient generator.

    Args:
        patient_generator (PatientGenerator): Patient Generator
        ds: LsadDataset
        frame_size (int): Frame size
        samples_per_patient (int | list[int], optional): # samples per patient. Defaults to 1.
        target_rate (int|None, optional): Target rate. Defaults to None.
        label_map (dict[int, int] | None, optional): Label map. Defaults to None.

    Returns:
        SampleGenerator: Sample generator

    Yields:
        Iterator[SampleGenerator]
    """

    return ds.signal_label_generator(
        patient_generator=patient_generator,
        frame_size=frame_size,
        samples_per_patient=samples_per_patient,
        target_rate=target_rate,
        label_map=lsad_label_map(label_map=label_map),
        label_type="scp",
        label_format=None,
    )
