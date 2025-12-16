from typing import Generator

import numpy.typing as npt
import helia_edge as helia

from ....datasets import PtbxlDataset, PtbxlScpCode, HKDataloader
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


class PtbxlDataloader(HKDataloader):
    def __init__(self, ds: PtbxlDataset, **kwargs):
        super().__init__(ds=ds, **kwargs)
        if self.label_map:
            self.label_map = {k: self.label_map[v] for (k, v) in PtbxlDiagnosticMap.items() if v in self.label_map}

        self.label_type = "scp"

    def data_generator(
        self,
        patient_ids: list[int],
        samples_per_patient: int | list[int],
        shuffle: bool = False,
    ) -> Generator[tuple[npt.NDArray, npt.NDArray], None, None]:
        return self.ds.signal_label_generator(
            patient_generator=helia.utils.uniform_id_generator(patient_ids, repeat=True, shuffle=shuffle),
            frame_size=self.frame_size,
            samples_per_patient=samples_per_patient,
            target_rate=self.sampling_rate,
            label_map=self.label_map,
            label_type=self.label_type,
            label_format="one_hot",
        )
