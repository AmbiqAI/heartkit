# Bring-Your-Own-Dataset (BYOD)

The Bring-Your-Own-Dataset (BYOD) feature allows users to add custom datasets for training and evaluating models. This feature is useful when working with proprietary or custom datasets that are not available in the heartKIT library.

## How it Works

1. **Create a Dataset**: Define a new dataset that inherits `HKDataset` and implements the required abstract methods.

```py linenums="1"

import numpy as np
import heartkit as hk

class MyDataset(hk.HKDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def name(self) -> str:
        return 'my-dataset'

    @property
    def sampling_rate(self) -> int:
        return 100

    def get_train_patient_ids(self) -> npt.NDArray:
        return np.arange(80)

    def get_test_patient_ids(self) -> npt.NDArray:
        return np.arange(80, 100)

    @contextlib.contextmanager
    def patient_data(self, patient_id: int) -> Generator[PatientData, None, None]:
        data = np.random.randn(1000)
        segs = np.random.randint(0, 1000, (10, 2))
        yield {"data": data, "segmentations": segs}

    def signal_generator(
        self,
        patient_generator: PatientGenerator,
        frame_size: int,
        samples_per_patient: int = 1,
        target_rate: int | None = None,
    ) -> Generator[npt.NDArray, None, None]:
        for patient in patient_generator:
            for _ in range(samples_per_patient):
                with self.patient_data(patient) as pt:
                    yield pt["data"]

    def download(self, num_workers: int | None = None, force: bool = False):
        pass

```

2. **Register the Dataset**: Register the new dataset with the `DatasetFactory` by calling the `register` method. This method takes the dataset name and the dataset class as arguments.

    ```py linenums="1"
    import heartkit as hk

    hk.DatasetFactory.register("my-dataset", CustomDataset)
    ```

3. **Use the Dataset**: The new dataset can now be used with the `DatasetFactory` to perform various operations such as downloading and generating data.

    ```py linenums="1"
    import heartkit as hk
    params = {}
    dataset = hk.DatasetFactory.get("my-dataset")(**params)
    ```
