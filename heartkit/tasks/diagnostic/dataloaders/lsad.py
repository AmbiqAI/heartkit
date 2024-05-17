from typing import Generator

import numpy.typing as npt

from ....datasets.defines import PatientGenerator
from ....datasets.lsad import LsadDataset, LsadScpCode
from ..defines import HKDiagnostic

LsadDiagnosticMap = {
    # NORM
    LsadScpCode.SR: HKDiagnostic.NORM,
    # STTC
    LsadScpCode.QTIE: HKDiagnostic.STTC,
    # LsadScpCode.STDD: HKDiagnostic.STTC,  # ?
    # LsadScpCode.STE: HKDiagnostic.STTC,  # ?
    LsadScpCode.STTC: HKDiagnostic.STTC,
    LsadScpCode.STTU: HKDiagnostic.STTC,
    # MI
    LsadScpCode.MI: HKDiagnostic.MI,
    LsadScpCode.AMI: HKDiagnostic.MI,
    LsadScpCode.AAMI: HKDiagnostic.MI,
    # HYP
    LsadScpCode.LVH: HKDiagnostic.HYP,
    LsadScpCode.LVH2: HKDiagnostic.HYP,
    LsadScpCode.RAH: HKDiagnostic.HYP,
    LsadScpCode.RVH: HKDiagnostic.HYP,
    LsadScpCode.LAH: HKDiagnostic.HYP,
    # CD
    LsadScpCode.AVB: HKDiagnostic.CD,
    LsadScpCode.AVB11: HKDiagnostic.CD,
    LsadScpCode.AVB2: HKDiagnostic.CD,
    LsadScpCode.AVB21: HKDiagnostic.CD,
    LsadScpCode.AVB22: HKDiagnostic.CD,
    LsadScpCode.AVB3: HKDiagnostic.CD,
    LsadScpCode.AVB221: HKDiagnostic.CD,
    LsadScpCode.BBB: HKDiagnostic.CD,
    LsadScpCode.LBBB: HKDiagnostic.CD,
    LsadScpCode.RBBB: HKDiagnostic.CD,
    LsadScpCode.ILBBB: HKDiagnostic.CD,
    LsadScpCode.CRBBB: HKDiagnostic.CD,
    LsadScpCode.CLBBB: HKDiagnostic.CD,
    LsadScpCode.IRBBB: HKDiagnostic.CD,
    LsadScpCode.IDC: HKDiagnostic.CD,
    LsadScpCode.AVD: HKDiagnostic.CD,
    LsadScpCode.WPW: HKDiagnostic.CD,
    LsadScpCode.LAFB: HKDiagnostic.CD,
    LsadScpCode.LPFB: HKDiagnostic.CD,
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
    return {k: label_map.get(v, -1) for (k, v) in LsadDiagnosticMap.items()}


def lsad_data_generator(
    patient_generator: PatientGenerator,
    ds: LsadDataset,
    frame_size: int,
    samples_per_patient: int | list[int] = 1,
    target_rate: int | None = None,
    label_map: dict[int, int] | None = None,
) -> Generator[tuple[npt.NDArray, npt.NDArray], None, None]:
    """Generate frames w/ diagnostic labels using patient generator.

    Args:
        patient_generator (PatientGenerator): Patient Generator
        ds: LsadDataset
        frame_size (int): Frame size
        samples_per_patient (int | list[int], optional): # samples per patient. Defaults to 1.
        target_rate (int|None, optional): Target rate. Defaults to None.
        label_map (dict[int, int] | None, optional): Label map. Defaults to None.

    Returns:
        Generator[tuple[npt.NDArray, npt.NDArray], None, None]: Sample generator

    """
    tgt_map = lsad_label_map(label_map=label_map)

    return ds.signal_label_generator(
        patient_generator=patient_generator,
        frame_size=frame_size,
        samples_per_patient=samples_per_patient,
        target_rate=target_rate,
        label_map=tgt_map,
        label_type="scp",
        label_format="multi_hot",
    )
