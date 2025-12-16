from typing import Generator

import numpy.typing as npt
import helia_edge as helia

from ....datasets import HKDataloader, PtbxlDataset, PtbxlScpCode
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


class PtbxlDataloader(HKDataloader):
    def __init__(self, ds: PtbxlDataset, **kwargs):
        super().__init__(ds=ds, **kwargs)
        if self.label_map:
            self.label_map = {k: self.label_map[v] for (k, v) in PtbxlRhythmMap.items() if v in self.label_map}
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
            label_format=None,
        )
