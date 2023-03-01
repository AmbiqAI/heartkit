import logging
import os

import numpy as np
import numpy.typing as npt
import sklearn
import tensorflow as tf

from ..defines import HeartTask
from ..tasks import get_num_classes, get_task_spec
from ..utils import load_pkl, save_pkl
from .defines import PatientGenerator, SampleGenerator
from .preprocess import preprocess_signal
from .utils import create_dataset_from_data

logger = logging.getLogger(__name__)


class EcgDataset:
    """ECG dataset base class"""

    ds_path: str
    task: HeartTask
    frame_size: int

    def __init__(self, ds_path: str, task: HeartTask = HeartTask.rhythm, frame_size: int = 1250) -> None:
        """ECG dataset base class"""
        self.ds_path = ds_path
        self.task = task
        self.frame_size = frame_size

    #############################################
    # !! Below must be implemented in subclass !!
    #############################################

    @property
    def sampling_rate(self) -> int:
        """Sampling rate in Hz"""
        return 0

    def get_train_patient_ids(self) -> npt.ArrayLike:
        """Get training patient IDs

        Returns:
            npt.ArrayLike: patient IDs
        """
        raise NotImplementedError()

    def get_test_patient_ids(self) -> npt.ArrayLike:
        """Get patient IDs reserved for testing only

        Returns:
            npt.ArrayLike: patient IDs
        """
        raise NotImplementedError()

    def task_data_generator(
        self,
        patient_generator: PatientGenerator,
        samples_per_patient: int | list[int] = 1,
    ) -> SampleGenerator:
        """Task-level data generator.

        Args:
            patient_generator (PatientGenerator): Patient data generator
            samples_per_patient (int | list[int], optional): # samples per patient. Defaults to 1.

        Returns:
            SampleGenerator: Sample data generator
        """
        raise NotImplementedError()

    def download(self, num_workers: int | None = None, force: bool = False):
        """Download dataset

        Args:
            num_workers (int | None, optional): # parallel workers. Defaults to None.
            force (bool, optional): Force redownload. Defaults to False.
        """
        raise NotImplementedError()

    def uniform_patient_generator(
        self,
        patient_ids: npt.ArrayLike,
        repeat: bool = True,
        shuffle: bool = True,
    ) -> PatientGenerator:
        """Yield data uniformly for each patient in the array.
        Args:
            patient_ids (pt.ArrayLike): Array of patient ids
            repeat (bool, optional): Whether to repeat generator. Defaults to True.
            shuffle (bool, optional): Whether to shuffle patient ids. Defaults to True.

        Returns:
            PatientGenerator: Patient generator

        Yields:
            Iterator[PatientGenerator]
        """
        raise NotImplementedError()

    #############################################
    # !! Above must be implemented in subclass !!
    #############################################

    def load_train_datasets(
        self,
        train_patients: float | None = None,
        val_patients: float | None = None,
        train_pt_samples: int | list[int] | None = None,
        val_pt_samples: int | list[int] | None = None,
        val_size: int | None = None,
        val_file: str | None = None,
        num_workers: int = 1,
    ) -> tuple[tf.data.Dataset, tf.data.Dataset]:
        """Load training and validation TF datasets
        Args:
            train_patients (float | None, optional): # or proportion of train patients. Defaults to None.
            val_patients (float | None, optional): # or proportion of val patients. Defaults to None.
            train_pt_samples (int | list[int] | None, optional): # samples per patient for training. Defaults to None.
            val_pt_samples (int | list[int] | None, optional): # samples per patient for validation. Defaults to None.
            val_size (int | None, optional): Validation size. Defaults to 200*len(val_patient_ids).
            val_file (str | None, optional): Path to existing pickled validation file. Defaults to None.
            num_workers (int, optional): # of parallel workers. Defaults to 1.

        Returns:
            tuple[tf.data.Dataset, tf.data.Dataset]: Training and validation datasets
        """

        if val_patients is not None and val_patients >= 1:
            val_patients = int(val_patients)

        train_pt_samples = train_pt_samples or 1000
        if val_pt_samples is None:
            val_pt_samples = train_pt_samples

        # Get train patients
        train_patient_ids = self.get_train_patient_ids()

        # Use subset of training patients
        if train_patients is not None:
            num_pts = int(train_patients) if train_patients > 1 else int(train_patients * len(train_patient_ids))
            train_patient_ids = train_patient_ids[:num_pts]
        # END IF

        # Use existing validation data
        if val_file and os.path.isfile(val_file):
            logger.info(f"Loading validation data from file {val_file}")
            val = load_pkl(val_file)
            val_ds = create_dataset_from_data(val["x"], val["y"], get_task_spec(self.task, self.frame_size))
            val_patient_ids = val["patient_ids"]
            train_patient_ids = np.setdiff1d(train_patient_ids, val_patient_ids)
        else:
            logger.info("Splitting patients into train and validation")
            train_patient_ids, val_patient_ids = self._split_train_test_patients(
                patient_ids=train_patient_ids, test_size=val_patients
            )
            if val_size is None:
                val_size = val_pt_samples * len(val_patient_ids)

            logger.info(f"Collecting {val_size} validation samples")
            val_ds = self._parallelize_dataset(
                patient_ids=val_patient_ids,
                samples_per_patient=val_pt_samples,
                repeat=False,
                num_workers=num_workers,
            )
            val_x, val_y = next(val_ds.batch(val_size).as_numpy_iterator())
            val_ds = create_dataset_from_data(val_x, val_y, get_task_spec(self.task, self.frame_size))

            # Cache validation set
            if val_file:
                logger.info(f"Caching the validation set in {val_file}")
                os.makedirs(os.path.dirname(val_file), exist_ok=True)
                save_pkl(val_file, x=val_x, y=val_y, patient_ids=val_patient_ids)
            # END IF
        # END IF

        logger.info("Building train dataset")
        train_ds = self._parallelize_dataset(
            patient_ids=train_patient_ids,
            samples_per_patient=train_pt_samples,
            repeat=True,
            num_workers=num_workers,
        )
        return train_ds, val_ds

    def load_test_dataset(
        self,
        test_patients: float | None = None,
        test_pt_samples: int | list[int] | None = None,
        num_workers: int = 1,
    ) -> tf.data.Dataset:
        """Load testing datasets
        Args:
            test_patients (float | None, optional): # or proportion of test patients. Defaults to None.
            test_pt_samples (int | None, optional): # samples per patient for testing. Defaults to None.
            num_workers (int, optional): # of parallel workers. Defaults to 1.

        Returns:
            tf.data.Dataset: Test dataset
        """
        test_patient_ids = self.get_test_patient_ids()

        if test_patients is not None:
            num_pts = int(test_patients) if test_patients > 1 else int(test_patients * len(test_patient_ids))
            test_patient_ids = test_patient_ids[:num_pts]
        # test_patient_ids = tf.convert_to_tensor(test_patient_ids)
        test_ds = self._parallelize_dataset(
            patient_ids=test_patient_ids,
            samples_per_patient=test_pt_samples,
            repeat=True,
            num_workers=num_workers,
        )
        return test_ds

    @tf.function
    def _parallelize_dataset(
        self,
        patient_ids: int = None,
        samples_per_patient: int | list[int] = 100,
        repeat: bool = False,
        num_workers: int = 1,
    ) -> tf.data.Dataset:
        """Generates datasets for given task in parallel using TF `interleave`

        Args:
            patient_ids (int, optional): List of patient IDs. Defaults to None.
            samples_per_patient (int, optional): # Samples per pateint. Defaults to 100.
            repeat (bool, optional): Should data generator repeat. Defaults to False.
            num_workers (int, optional): Number of parallel workers. Defaults to 1.
        Returns:
            tf.data.Dataset: Parallelize dataset
        """

        def _make_train_dataset(i, split):
            return self._create_dataset_from_generator(
                patient_ids=patient_ids[i * split : (i + 1) * split],
                samples_per_patient=samples_per_patient,
                repeat=repeat,
            )

        split = len(patient_ids) // num_workers
        datasets = [_make_train_dataset(i, split) for i in range(num_workers)]
        if num_workers <= 1:
            return datasets[0]

        return tf.data.Dataset.from_tensor_slices(datasets).interleave(
            lambda x: x,
            cycle_length=num_workers,
            num_parallel_calls=tf.data.AUTOTUNE,
        )

    def _split_train_test_patients(self, patient_ids: npt.ArrayLike, test_size: float) -> list[list[int]]:
        """Perform train/test split on patients for given task.

        Args:
            patient_ids (npt.ArrayLike): Patient Ids
            test_size (float): Test size

        Returns:
            list[list[int]]: Train and test sets of patient ids
        """
        return sklearn.model_selection.train_test_split(patient_ids, test_size=test_size)

    def _create_dataset_from_generator(
        self,
        patient_ids: npt.ArrayLike,
        samples_per_patient: int | list[int] = 1,
        repeat: bool = True,
    ) -> tf.data.Dataset:
        """Creates TF dataset generator for task.

        Args:
            patient_ids (npt.ArrayLike): Patient IDs
            samples_per_patient (int | list[int], optional): Samples per patient. Defaults to 1.
            repeat (bool, optional): Repeat. Defaults to True.

        Returns:
            tf.data.Dataset: Dataset generator
        """
        dataset = tf.data.Dataset.from_generator(
            generator=self._dataset_sample_generator,
            output_signature=get_task_spec(self.task, self.frame_size),
            args=(patient_ids, samples_per_patient, repeat),
        )
        return dataset

    def _dataset_sample_generator(
        self,
        patient_ids: npt.ArrayLike,
        samples_per_patient: int | list[int] = 1,
        repeat: bool = True,
    ) -> SampleGenerator:
        """Internal sample generator for task.

        Args:
            patient_ids (npt.ArrayLike): Patient IDs
            samples_per_patient (int | list[int], optional): Samples per patient. Defaults to 1.
            repeat (bool, optional): Repeat. Defaults to True.

        Returns:
            SampleGenerator: Task sample generator
        """
        num_classes = get_num_classes(self.task)
        patient_generator = self.uniform_patient_generator(patient_ids, repeat=repeat)
        data_generator = self.task_data_generator(
            patient_generator,
            samples_per_patient=samples_per_patient,
        )
        data_generator = map(
            lambda x_y: (
                # Apply augmentations
                # Pre-process signal and convert to 2D shape
                preprocess_signal(data=x_y[0], sample_rate=self.sampling_rate, target_rate=None).reshape((1, -1, 1)),
                tf.one_hot(x_y[1], num_classes),  # x_y[1]
            ),
            data_generator,
        )

        return data_generator
