from typing import Generator

import numpy.typing as npt
import helia_edge as helia

from ....datasets import HKDataloader, LsadDataset, LsadScpCode
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


class LsadDataloader(HKDataloader):
    def __init__(self, ds: LsadDataset, **kwargs):
        super().__init__(ds=ds, **kwargs)
        if self.label_map:
            self.label_map = {k: self.label_map[v] for (k, v) in LsadRhythmMap.items() if v in self.label_map}
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
