from typing import Generator

import numpy.typing as npt
import helia_edge as helia

from ....datasets import LsadDataset, LsadScpCode, HKDataloader
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


class LsadDataloader(HKDataloader):
    def __init__(self, ds: LsadDataset, **kwargs):
        super().__init__(ds=ds, **kwargs)
        if self.label_map:
            self.label_map = {k: self.label_map[v] for (k, v) in LsadDiagnosticMap.items() if v in self.label_map}

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
            label_format="multi_hot",
        )
