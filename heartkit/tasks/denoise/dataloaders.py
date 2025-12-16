from typing import Generator

import numpy as np
import numpy.typing as npt
import helia_edge as helia


from ...datasets import HKDataloader


class DenoiseDataloader(HKDataloader):
    def __init__(self, **kwargs):
        """Generic Dataloader for denoising task."""
        super().__init__(**kwargs)

    def data_generator(
        self,
        patient_ids: list[int],
        samples_per_patient: int | list[int],
        shuffle: bool = False,
    ) -> Generator[npt.NDArray, None, None]:
        """Generate data for given patient ids.
        Leveraging the signal_generator method from the dataset class to generate data.
        """
        gen = self.ds.signal_generator(
            patient_generator=helia.utils.uniform_id_generator(patient_ids, repeat=True, shuffle=shuffle),
            frame_size=self.frame_size,
            samples_per_patient=samples_per_patient,
            target_rate=self.sampling_rate,
        )
        for x in gen:
            x = np.nan_to_num(x, neginf=0, posinf=0).astype(np.float32)
            x = np.reshape(x, (-1, 1))
            yield x


DenoiseTaskFactory = helia.utils.create_factory(factory="HKDenoiseTaskFactory", type=HKDataloader)
DenoiseTaskFactory.register("ecg-synthetic", DenoiseDataloader)
DenoiseTaskFactory.register("ppg-synthetic", DenoiseDataloader)
DenoiseTaskFactory.register("icentia11k", DenoiseDataloader)
DenoiseTaskFactory.register("icentia_mini", DenoiseDataloader)
DenoiseTaskFactory.register("ptbxl", DenoiseDataloader)
DenoiseTaskFactory.register("lsad", DenoiseDataloader)
DenoiseTaskFactory.register("qtdb", DenoiseDataloader)
