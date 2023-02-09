import logging
import os
from typing import List, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt
import sklearn
import tensorflow as tf

from ..types import EcgTask
from ..utils import load_pkl, save_pkl
from .types import PatientGenerator, SampleGenerator
from .utils import create_dataset_from_data, get_task_spec

logger = logging.getLogger(__name__)


class EcgDataset:
    """ECG dataset base class"""

    ds_path: str
    task: EcgTask
    frame_size: int

    def __init__(
        self, ds_path: str, task: EcgTask = EcgTask.rhythm, frame_size: int = 1250
    ) -> None:
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

    @property
    def mean(self) -> float:
        """Dataset mean"""
        return 0

    @property
    def std(self) -> float:
        """Dataset st dev"""
        return 1

    def get_train_patient_ids(self) -> npt.ArrayLike:
        """Get list of training patient Ids."""
        raise NotImplementedError()

    def get_test_patient_ids(self) -> npt.ArrayLike:
        """Get list of test patient Ids."""
        raise NotImplementedError()

    def task_data_generator(
        self,
        patient_generator: PatientGenerator,
        samples_per_patient: Union[int, List[int]] = 1,
    ) -> SampleGenerator:
        """Task data generator."""
        raise NotImplementedError()

    def download(self, num_workers: Optional[int] = None, force: bool = False):
        """Download dataset."""
        raise NotImplementedError()

    def uniform_patient_generator(
        self,
        patient_ids: npt.ArrayLike,
        repeat: bool = True,
        shuffle: bool = True,
    ) -> PatientGenerator:
        """Yield data for each patient in the array.
        Args:
            patient_ids (pt.ArrayLike): Array of patient ids
            repeat (bool, optional): Whether to repeat generator. Defaults to True.
            shuffle (bool, optional): Whether to shuffle patient ids.. Defaults to True.

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
        train_patients: Optional[float] = None,
        val_patients: Optional[float] = None,
        train_pt_samples: Optional[Union[int, List[int]]] = None,
        val_pt_samples: Optional[Union[int, List[int]]] = None,
        val_size: Optional[int] = None,
        val_file: Optional[str] = None,
        num_workers: int = 1,
    ) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
        """Load training and validation TF datasets
        Args:
            train_patients (Optional[float], optional): # or proportion of train patients. Defaults to None.
            val_patients (Optional[float], optional): # or proportion of val patients. Defaults to None.
            train_pt_samples (Optional[Union[int, List[int]]], optional): # samples per patient for training. Defaults to None.
            val_pt_samples (Optional[Union[int, List[int]]], optional): # samples per patient for validation. Defaults to None.
            val_file (Optional[str], optional): Path to existing pickled validation file. Defaults to None.
            num_workers (int, optional): # of parallel workers. Defaults to 1.

        Returns:
            Tuple[tf.data.Dataset, tf.data.Dataset]: Training and validation datasets
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
            num_pts = (
                int(train_patients)
                if train_patients > 1
                else int(train_patients * len(train_patient_ids))
            )
            train_patient_ids = train_patient_ids[:num_pts]
        # END IF

        # Use existing validation data
        if val_file and os.path.isfile(val_file):
            logger.info(f"Loading validation data from file {val_file}")
            val = load_pkl(val_file)
            val_ds = create_dataset_from_data(
                val["x"], val["y"], task=self.task, frame_size=self.frame_size
            )
            val_patient_ids = val["patient_ids"]
            train_patient_ids = np.setdiff1d(train_patient_ids, val_patient_ids)
        else:
            logger.info("Splitting patients into train and validation")
            train_patient_ids, val_patient_ids = self._split_train_test_patients(
                patient_ids=train_patient_ids, test_size=val_patients
            )
            if val_size is None:
                val_size = 200 * len(val_patient_ids)

            logger.info(f"Collecting {val_size} validation samples")
            val_ds = self.parallelize_dataset(
                patient_ids=val_patient_ids,
                samples_per_patient=val_pt_samples,
                repeat=False,
                num_workers=num_workers,
            )
            val_x, val_y = next(val_ds.batch(val_size).as_numpy_iterator())
            val_ds = create_dataset_from_data(
                val_x, val_y, task=self.task, frame_size=self.frame_size
            )

            # Cache validation set
            if val_file:
                logger.info(f"Caching the validation set in {val_file}")
                os.makedirs(os.path.dirname(val_file), exist_ok=True)
                save_pkl(val_file, x=val_x, y=val_y, patient_ids=val_patient_ids)
            # END IF
        # END IF

        logger.info("Building train dataset")
        train_ds = self.parallelize_dataset(
            patient_ids=train_patient_ids,
            samples_per_patient=train_pt_samples,
            repeat=True,
            num_workers=num_workers,
        )
        return train_ds, val_ds

    def load_test_dataset(
        self,
        test_patients: Optional[float] = None,
        test_pt_samples: Optional[Union[int, List[int]]] = None,
        num_workers: int = 1,
    ) -> tf.data.Dataset:
        """Load testing datasets
        Args:
            test_patients (Optional[float], optional): # or proportion of test patients. Defaults to None.
            test_pt_samples (Optional[int], optional): # samples per patient for testing. Defaults to None.
            num_workers (int, optional): # of parallel workers. Defaults to 1.

        Returns:
            tf.data.Dataset: Test dataset
        """
        test_patient_ids = self.get_test_patient_ids()

        if test_patients is not None:
            num_pts = (
                int(test_patients)
                if test_patients > 1
                else int(test_patients * len(test_patient_ids))
            )
            test_patient_ids = test_patient_ids[:num_pts]
        test_patient_ids = tf.convert_to_tensor(test_patient_ids)
        test_ds = self.parallelize_dataset(
            patient_ids=test_patient_ids,
            samples_per_patient=test_pt_samples,
            repeat=True,
            num_workers=num_workers,
        )
        return test_ds

    @tf.function
    def parallelize_dataset(
        self,
        patient_ids: int = None,
        samples_per_patient: Union[int, List[int]] = 100,
        repeat: bool = False,
        num_workers: int = 1,
    ) -> tf.data.Dataset:
        """Generates datasets for given task in parallel using TF `interleave`

        Args:
            ds_path (str): Dataset path
            task (EcgTask, optional): ECG Task routine.
            patient_ids (int, optional): List of patient IDs. Defaults to None.
            frame_size (int, optional): Frame size. Defaults to 1250.
            samples_per_patient (int, optional): # Samples per pateint. Defaults to 100.
            repeat (bool, optional): Should data generator repeat. Defaults to False.
            num_workers (int, optional): Number of parallel workers. Defaults to 1.
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
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
        )

    def _split_train_test_patients(
        self, patient_ids: npt.ArrayLike, test_size: float
    ) -> List[List[int]]:
        """Perform train/test split on patients for given task.

        Args:
            task (EcgTask): Heart task
            patient_ids (npt.ArrayLike): Patient Ids
            test_size (float): Test size

        Returns:
            List[List[int]]: Train and test sets of patient ids
        """
        return sklearn.model_selection.train_test_split(
            patient_ids, test_size=test_size
        )

    def _create_dataset_from_generator(
        self,
        patient_ids: npt.ArrayLike,
        samples_per_patient: Union[int, List[int]] = 1,
        repeat: bool = True,
    ) -> tf.data.Dataset:
        """Creates TF dataset generator for task.

        Args:
            patient_ids (npt.ArrayLike): Patient IDs
            samples_per_patient (Union[int, List[int]], optional): Samples per patient. Defaults to 1.
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
        samples_per_patient: Union[int, List[int]] = 1,
        repeat: bool = True,
    ) -> SampleGenerator:
        """Internal sample generator for task.

        Args:
            patient_ids (npt.ArrayLike): Patient IDs
            samples_per_patient (Union[int, List[int]], optional): Samples per patient. Defaults to 1.
            repeat (bool, optional): Repeat. Defaults to True.

        Returns:
            SampleGenerator: Task sample generator
        """
        patient_generator = self.uniform_patient_generator(patient_ids, repeat=repeat)
        data_generator = self.task_data_generator(
            patient_generator,
            samples_per_patient=samples_per_patient,
        )
        return data_generator
