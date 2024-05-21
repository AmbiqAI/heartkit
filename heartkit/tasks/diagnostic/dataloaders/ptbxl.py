from typing import Generator

import numpy.typing as npt

from ....datasets.defines import PatientGenerator
from ....datasets.ptbxl import PtbxlDataset, PtbxlScpCode
from ..defines import HKDiagnostic

PtbxlDiagnosticMap = {
    # NORM
    PtbxlScpCode.NORM: HKDiagnostic.NORM,
    # STTC
    # PtbxlScpCode.NDT: HKDiagnostic.STTC, FORM?
    # PtbxlScpCode.NST_: HKDiagnostic.STTC, FORM?
    # PtbxlScpCode.DIG: HKDiagnostic.STTC, FORM?
    # PtbxlScpCode.LNGQT: HKDiagnostic.STTC, FORM?
    PtbxlScpCode.ISC_: HKDiagnostic.STTC,
    PtbxlScpCode.ISCAL: HKDiagnostic.STTC,
    PtbxlScpCode.ISCIN: HKDiagnostic.STTC,
    PtbxlScpCode.ISCIL: HKDiagnostic.STTC,
    PtbxlScpCode.ISCAS: HKDiagnostic.STTC,
    PtbxlScpCode.ISCLA: HKDiagnostic.STTC,
    PtbxlScpCode.ANEUR: HKDiagnostic.STTC,
    PtbxlScpCode.EL: HKDiagnostic.STTC,
    PtbxlScpCode.ISCAN: HKDiagnostic.STTC,
    # MI
    PtbxlScpCode.IMI: HKDiagnostic.MI,
    PtbxlScpCode.ASMI: HKDiagnostic.MI,
    PtbxlScpCode.ILMI: HKDiagnostic.MI,
    PtbxlScpCode.AMI: HKDiagnostic.MI,
    PtbxlScpCode.ALMI: HKDiagnostic.MI,
    PtbxlScpCode.INJAS: HKDiagnostic.MI,
    PtbxlScpCode.LMI: HKDiagnostic.MI,
    PtbxlScpCode.INJAL: HKDiagnostic.MI,
    PtbxlScpCode.IPLMI: HKDiagnostic.MI,
    PtbxlScpCode.IPMI: HKDiagnostic.MI,
    PtbxlScpCode.INJIN: HKDiagnostic.MI,
    PtbxlScpCode.INJLA: HKDiagnostic.MI,
    PtbxlScpCode.PMI: HKDiagnostic.MI,
    PtbxlScpCode.INJIL: HKDiagnostic.MI,
    # HYP
    PtbxlScpCode.LVH: HKDiagnostic.HYP,
    PtbxlScpCode.LAO_LAE: HKDiagnostic.HYP,
    PtbxlScpCode.RVH: HKDiagnostic.HYP,
    PtbxlScpCode.RAO_RAE: HKDiagnostic.HYP,
    PtbxlScpCode.SEHYP: HKDiagnostic.HYP,
    # CD
    PtbxlScpCode.LAFB: HKDiagnostic.CD,
    PtbxlScpCode.IRBBB: HKDiagnostic.CD,
    PtbxlScpCode.AVB1: HKDiagnostic.CD,
    PtbxlScpCode.IVCD: HKDiagnostic.CD,
    PtbxlScpCode.CRBBB: HKDiagnostic.CD,
    PtbxlScpCode.CLBBB: HKDiagnostic.CD,
    PtbxlScpCode.LPFB: HKDiagnostic.CD,
    PtbxlScpCode.WPW: HKDiagnostic.CD,
    PtbxlScpCode.ILBBB: HKDiagnostic.CD,
    PtbxlScpCode.AVB2: HKDiagnostic.CD,
    PtbxlScpCode.AVB3: HKDiagnostic.CD,
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
    return {k: label_map.get(v, -1) for (k, v) in PtbxlDiagnosticMap.items()}


def ptbxl_data_generator(
    patient_generator: PatientGenerator,
    ds: PtbxlDataset,
    frame_size: int,
    samples_per_patient: int | list[int] = 1,
    target_rate: int | None = None,
    label_map: dict[int, int] | None = None,
) -> Generator[tuple[npt.NDArray, npt.NDArray], None, None]:
    """Generate frames w/ diagnostic labels using patient generator.

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
    tgt_map = ptbxl_label_map(label_map=label_map)

    return ds.signal_label_generator(
        patient_generator=patient_generator,
        frame_size=frame_size,
        samples_per_patient=samples_per_patient,
        target_rate=target_rate,
        label_map=tgt_map,
        label_type="scp",
        label_format="multi_hot",
    )
