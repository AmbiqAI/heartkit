import functools
import logging
from typing import Generator
from collections.abc import Iterable

import numpy as np
import numpy.typing as npt
import tensorflow as tf
import helia_edge as helia


from .dataset import HKDataset

logger = logging.getLogger(__name__)


class HKDataloader:
    ds: HKDataset
    frame_size: int
    sampling_rate: int
    label_map: dict[int, int] | None
    label_type: str | None

    def __init__(
        self,
        ds: HKDataset,
        frame_size: int = 1000,
        sampling_rate: int = 100,
        label_map: dict[int, int] | None = None,
        label_type: str | None = None,
        **kwargs,
    ):
        """HKDataloader is used to create a task specific dataloader for a dataset.
        This class should be subclassed for specific task and dataset. If multiple datasets are needed for given task,
        multiple dataloaders can be created. To simplify the process, the dataloaders can be placed in an ItemFactory.

        Args:
            ds (HKDataset): Dataset
            frame_size (int, optional): Frame size. Defaults to 1000.
            sampling_rate (int, optional): Sampling rate. Defaults to 100.
            label_map (dict[int, int], optional): Label map. Defaults to None.
            label_type (str, optional): Label type. Defaults to None.

        Example:

        ```python
        from typing import Generator
        import numpy as np
        import numpy.typing as npt
        import heartkit as hk

        class MyDataloader(hk.HKDataloader):
            def __init__(self, ds: hk.HKDataset, **kwargs):
                super().__init__(ds=ds, **kwargs)

            def patient_generator(
                self,
                patient_id: int,
                samples_per_patient: list[int],
            ) -> Generator[npt.NDArray, None, None]:

                # Implement patient generator
                with ds.patient_data(patient_id) as pt:
                    for _ in range(samples_per_patient):
                        data = pt["data"][:]
                        # Grab random frame and lead
                        lead = np.random.randint(0, data.shape[0])
                        start = np.random.randint(0, data.shape[1] - self.frame_size)
                        frame = data[lead, start : start + self.frame_size]
                        yield frame

            def data_generator(
                self,
                patient_ids: list[int],
                samples_per_patient: int | list[int],
                shuffle: bool = False,
            ) -> Generator[npt.NDArray, None, None]:
                for pt_id in helia.utils.uniform_id_generator(patient_ids, shuffle=shuffle):
                    # Implement data generator
                    yield data
                # END FOR
        ```

        """
        self.ds = ds
        self.frame_size = frame_size
        self.sampling_rate = sampling_rate
        self.label_map = label_map
        self.label_type = label_type

    def split_train_val_patients(
        self,
        train_patients: list[int] | float | None = None,
        val_patients: list[int] | float | None = None,
    ) -> tuple[list[int], list[int]]:
        """Split patients into training and validation sets. Unless train_patients or
        val_patients are provided, the default is to call the dataset's split_train_test_patients

        Args:
            train_patients (list[int] | float | None, optional): Training patients. Defaults to None.
            val_patients (list[int] | float | None, optional): Validation patients. Defaults to None.

        Returns:
            tuple[list[int], list[int]]: Training and validation patient ids
        """
        # Get train patients
        train_patient_ids = self.ds.get_train_patient_ids()
        train_patient_ids = self.ds.filter_patients_for_labels(
            patient_ids=train_patient_ids,
            label_map=self.label_map,
            label_type=self.label_type,
        )

        # Use subset of training patients
        if isinstance(train_patients, Iterable):
            train_patient_ids = train_patients

        if train_patients is not None:
            num_pts = int(train_patients) if train_patients > 1 else int(train_patients * len(train_patient_ids))
            train_patient_ids = train_patient_ids[:num_pts]
            logger.debug(f"Using {len(train_patient_ids)} training patients")
        # END IF

        # Use subset of validation patients
        if isinstance(val_patients, Iterable):
            val_patient_ids = val_patients
            train_patient_ids = np.setdiff1d(train_patient_ids, val_patient_ids).tolist()
            return train_patient_ids, val_patient_ids

        if val_patients is not None and val_patients >= 1:
            val_patients = int(val_patients)

        train_patient_ids, val_patient_ids = self.ds.split_train_test_patients(
            patient_ids=train_patient_ids,
            test_size=val_patients,
            label_map=self.label_map,
            label_type=self.label_type,
        )

        return train_patient_ids, val_patient_ids

    def test_patient_ids(
        self,
        test_patients: float | None = None,
    ) -> list[int]:
        """Get test patient ids

        Args:
            test_patients (float | None, optional): Test patients. Defaults to None.

        Returns:
            list[int]: Test patient ids
        """
        test_patient_ids = self.ds.get_test_patient_ids()
        test_patient_ids = self.ds.filter_patients_for_labels(
            patient_ids=test_patient_ids,
            label_map=self.label_map,
            label_type=self.label_type,
        )

        if test_patients is not None:
            num_pts = int(test_patients) if test_patients > 1 else int(test_patients * len(test_patient_ids))
            test_patient_ids = test_patient_ids[:num_pts]

        return test_patient_ids

    def patient_data_generator(
        self,
        patient_id: int,
        samples_per_patient: list[int],
    ) -> Generator[tuple[npt.NDArray, ...], None, None]:
        """Generate data for given patient id

        Args:
            patient_id (int): Patient ID
            samples_per_patient (list[int]): Samples per patient

        Returns:
            Generator[tuple[npt.NDArray, ...], None, None]: Data generator


        !!! note
            This method should be implemented in the subclass

        """
        raise NotImplementedError()

    def data_generator(
        self,
        patient_ids: list[int],
        samples_per_patient: int | list[int],
        shuffle: bool = False,
    ) -> Generator[tuple[npt.NDArray, ...], None, None]:
        """Generate data for given patient ids

        Args:
            patient_ids (list[int]): Patient IDs
            samples_per_patient (int | list[int]): Samples per patient
            shuffle (bool, optional): Shuffle data. Defaults to False.

        Returns:
            Generator[tuple[npt.NDArray, ...], None, None]: Data generator

        """
        for pt_id in helia.utils.uniform_id_generator(patient_ids, shuffle=shuffle):
            for data in self.patient_data_generator(pt_id, samples_per_patient):
                yield data
            # END FOR
        # END FOR

    def create_dataloader(
        self, patient_ids: list[int], samples_per_patient: int | list[int], shuffle: bool = False
    ) -> tf.data.Dataset:
        """Create tf.data.Dataset from internal data generator

        Args:
            patient_ids (list[int]): Patient IDs
            samples_per_patient (int | list[int]): Samples per patient
            shuffle (bool, optional): Shuffle data. Defaults to False.

        Returns:
            tf.data.Dataset: Dataset
        """
        data_gen = functools.partial(
            self.data_generator,
            patient_ids=patient_ids,
            samples_per_patient=samples_per_patient,
            shuffle=shuffle,
        )

        # Compute output signature from generator
        sig = helia.utils.get_output_signature_from_gen(data_gen)

        ds = tf.data.Dataset.from_generator(
            data_gen,
            output_signature=sig,
        )
        return ds
