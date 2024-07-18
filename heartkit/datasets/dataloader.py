import functools
import logging
import math
import os
from typing import Callable, Generator

import numpy as np
import numpy.typing as npt
import tensorflow as tf

from ..utils import load_pkl, save_pkl
from .dataset import HKDataset
from .defines import PatientGenerator, Preprocessor
from .utils import (
    create_dataset_from_data,
    create_interleaved_dataset_from_generator,
    uniform_id_generator,
)

logger = logging.getLogger(__name__)


def train_val_dataloader(
    ds: HKDataset,
    spec: tuple[tf.TensorSpec, tf.TensorSpec],
    data_generator: Callable[
        [PatientGenerator, int | list[int]], Generator[tuple[npt.NDArray, npt.NDArray], None, None]
    ],
    id_generator: PatientGenerator | None = None,
    train_patients: float | None = None,
    val_patients: float | None = None,
    val_pt_samples: int | None = None,
    val_file: os.PathLike | None = None,
    val_size: int | None = None,
    label_map: dict[int, int] | None = None,
    label_type: str | None = None,
    preprocess: Preprocessor | None = None,
    val_preprocess: Preprocessor | None = None,
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

    if id_generator is None:
        id_generator = functools.partial(uniform_id_generator, repeat=True)

    if val_patients is not None and val_patients >= 1:
        val_patients = int(val_patients)

    if val_preprocess is None:
        val_preprocess = preprocess

    val_pt_samples = val_pt_samples or 100

    # Get train patients
    train_patient_ids = ds.get_train_patient_ids()
    train_patient_ids = ds.filter_patients_for_labels(
        patient_ids=train_patient_ids,
        label_map=label_map,
        label_type=label_type,
    )

    # Use subset of training patients
    if train_patients is not None:
        num_pts = int(train_patients) if train_patients > 1 else int(train_patients * len(train_patient_ids))
        train_patient_ids = train_patient_ids[:num_pts]
        logger.debug(f"Using {len(train_patient_ids)} training patients")
    # END IF

    if ds.cachable and val_file and os.path.isfile(val_file):
        logger.debug(f"Loading validation data from file {val_file}")
        val = load_pkl(val_file)
        val_patient_ids = val["patient_ids"]
        train_patient_ids = np.setdiff1d(train_patient_ids, val_patient_ids)
        val_ds = create_dataset_from_data(val["x"], val["y"], spec)

    else:
        logger.debug("Splitting patients into train and validation")
        train_patient_ids, val_patient_ids = ds.split_train_test_patients(
            patient_ids=train_patient_ids,
            test_size=val_patients,
            label_map=label_map,
            label_type=label_type,
        )
        if val_size is None:
            num_samples = np.mean(val_pt_samples) if isinstance(val_pt_samples, list) else val_pt_samples
            val_size = math.ceil(num_samples * len(val_patient_ids))

        logger.debug(f"Collecting {val_size} validation samples")

        val_ds = create_interleaved_dataset_from_generator(
            data_generator=data_generator,
            id_generator=id_generator,
            ids=val_patient_ids,
            spec=spec,
            preprocess=val_preprocess,
            num_workers=num_workers,
        )

        val_x, val_y = next(val_ds.batch(val_size).as_numpy_iterator())
        val_ds = create_dataset_from_data(val_x, val_y, spec)

        # Cache validation set
        if ds.cachable and val_file:
            logger.debug(f"Caching the validation set in {val_file}")
            os.makedirs(os.path.dirname(val_file), exist_ok=True)
            save_pkl(val_file, x=val_x, y=val_y, patient_ids=val_patient_ids)
        # END IF
    # END IF

    logger.debug("Building train dataset")

    train_ds = create_interleaved_dataset_from_generator(
        data_generator=data_generator,
        id_generator=id_generator,
        ids=train_patient_ids,
        spec=spec,
        preprocess=preprocess,
        num_workers=num_workers,
    )

    return train_ds, val_ds


def test_dataloader(
    ds: HKDataset,
    spec: tuple[tf.TensorSpec, tf.TensorSpec],
    data_generator: Callable[
        [PatientGenerator, int | list[int]], Generator[tuple[npt.NDArray, npt.NDArray], None, None]
    ],
    id_generator: PatientGenerator | None = None,
    test_patients: float | None = None,
    test_file: os.PathLike | None = None,
    label_map: dict[int, int] | None = None,
    label_type: str | None = None,
    preprocess: Preprocessor | None = None,
    num_workers: int = 1,
) -> tf.data.Dataset:
    """Load testing datasets

    Args:
        test_patients (float | None, optional): # or proportion of test patients. Defaults to None.
        test_pt_samples (int | None, optional): # samples per patient for testing. Defaults to None.
        test_file (str | None, optional): Path to existing pickled test file. Defaults to None.
        repeat (bool, optional): Restart generator when dataset is exhausted. Defaults to True.
        num_workers (int, optional): # of parallel workers. Defaults to 1.

    Returns:
        tf.data.Dataset: Test dataset
    """

    # Get test patients
    test_patient_ids = ds.get_test_patient_ids()
    test_patient_ids = ds.filter_patients_for_labels(
        patient_ids=test_patient_ids,
        label_map=label_map,
        label_type=label_type,
    )

    if test_patients is not None:
        num_pts = int(test_patients) if test_patients > 1 else int(test_patients * len(test_patient_ids))
        test_patient_ids = test_patient_ids[:num_pts]

    # Use existing validation data
    if ds.cachable and test_file and os.path.isfile(test_file):
        logger.debug(f"Loading test data from file {test_file}")
        test = load_pkl(test_file)
        test_ds = create_dataset_from_data(test["x"], test["y"], spec)
        test_patient_ids = test["patient_ids"]
    else:
        test_ds = create_interleaved_dataset_from_generator(
            data_generator=data_generator,
            id_generator=id_generator,
            ids=test_patient_ids,
            spec=spec,
            preprocess=preprocess,
            num_workers=num_workers,
        )

    return test_ds
