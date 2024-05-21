from typing import Generator

import numpy.typing as npt

from ....datasets.defines import PatientGenerator
from ....datasets.ptbxl import PtbxlDataset, PtbxlScpCode
from ..defines import HKRhythm

PtbxlRhythmMap = {
    PtbxlScpCode.SR: HKRhythm.sr,
    PtbxlScpCode.AFIB: HKRhythm.afib,
    PtbxlScpCode.AFLT: HKRhythm.aflut,
    PtbxlScpCode.STACH: HKRhythm.stach,
    PtbxlScpCode.SBRAD: HKRhythm.sbrad,
    PtbxlScpCode.SARRH: HKRhythm.sarrh,
    PtbxlScpCode.SVARR: HKRhythm.svarr,
    PtbxlScpCode.SVTAC: HKRhythm.svt,
    PtbxlScpCode.PSVT: HKRhythm.svt,
    PtbxlScpCode.BIGU: HKRhythm.bigu,
    PtbxlScpCode.TRIGU: HKRhythm.trigu,
    PtbxlScpCode.PACE: HKRhythm.pace,
}


def ptbxl_label_map(
    label_map: dict[int, int] | None = None,
) -> dict[int, int]:
    """Get label map

    Args:
        label_map (dict[int, int]|None): Label map

    Returns:
        dict[int, int]: Label map
    """
    return {k: label_map.get(v, -1) for (k, v) in PtbxlRhythmMap.items()}


def ptbxl_data_generator(
    patient_generator: PatientGenerator,
    ds: PtbxlDataset,
    frame_size: int,
    samples_per_patient: int | list[int] = 1,
    target_rate: int | None = None,
    label_map: dict[int, int] | None = None,
) -> Generator[tuple[npt.NDArray, int], None, None]:
    """Generate frames w/ rhythm labels using patient generator.

    Args:
        patient_generator (PatientGenerator): Patient Generator
        ds: PtbxlDataset
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
        label_map=ptbxl_label_map(label_map=label_map),
        label_type="scp",
        label_format=None,
    )
