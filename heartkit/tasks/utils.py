"""
# Task Utils API

Utility functions for tasks

Functions:
    load_train_dataloader_split: Load training and validation dataloader pipeline
    load_test_dataloader_split: Load test dataloader pipeline

"""

import numpy as np
import tensorflow as tf
import helia_edge as helia

from ..datasets import HKDataset, HKDataloader
from ..defines import HKTaskParams

logger = helia.utils.setup_logger(__name__)


def load_train_dataloader_split(
    datasets: list[HKDataset],
    params: HKTaskParams,
    factory: helia.utils.ItemFactory[HKDataloader],
) -> tuple[tf.data.Dataset, tf.data.Dataset]:
    """Create dataloader pipeline for training and validation sets

    Args:
        datasets (list[HKDataset]): List of datasets
        params (HKTaskParams): Training parameters
    """
    train_datasets = []
    val_datasets = []
    for ds in datasets:
        dataloader = factory.get(ds.name)(
            ds=ds,
            frame_size=params.frame_size,
            sampling_rate=params.sampling_rate,
            label_map=params.class_map,
        )
        train_patients, val_patients = dataloader.split_train_val_patients(
            train_patients=params.train_patients,
            val_patients=params.val_patients,
        )

        train_ds = dataloader.create_dataloader(
            patient_ids=train_patients, samples_per_patient=params.samples_per_patient, shuffle=True
        )

        val_ds = dataloader.create_dataloader(
            patient_ids=val_patients, samples_per_patient=params.val_samples_per_patient, shuffle=False
        )
        train_datasets.append(train_ds)
        val_datasets.append(val_ds)
    # END FOR

    ds_weights = None
    if params.dataset_weights:
        ds_weights = np.array(params.dataset_weights)
        ds_weights = ds_weights / ds_weights.sum()

    train_ds = tf.data.Dataset.sample_from_datasets(train_datasets, weights=ds_weights)
    val_ds = tf.data.Dataset.sample_from_datasets(val_datasets, weights=ds_weights)

    return train_ds, val_ds


def load_test_dataloader_split(
    datasets: list[HKDataset],
    params: HKTaskParams,
    factory: helia.utils.ItemFactory[HKDataloader],
) -> tf.data.Dataset:
    """Create dataloader pipeline for test set

    Args:
        datasets (list[HKDataset]): List of datasets
        params (HKTaskParams): Test or export parameters

    Returns:
        tf.data.Dataset: Test dataset pipeline
    """
    test_datasets = []
    for ds in datasets:
        dataloader = factory.get(ds.name)(
            ds=ds,
            frame_size=params.frame_size,
            sampling_rate=params.sampling_rate,
            label_map=params.class_map,
        )
        test_patients = dataloader.test_patient_ids(params.test_patients)
        test_ds = dataloader.create_dataloader(
            patient_ids=test_patients,
            samples_per_patient=params.test_samples_per_patient,
            shuffle=False,
        )
        test_datasets.append(test_ds)
    # END FOR

    ds_weights = None
    if params.dataset_weights:
        ds_weights = np.array(params.dataset_weights)
        ds_weights = ds_weights / ds_weights.sum()

    test_ds = tf.data.Dataset.sample_from_datasets(test_datasets, weights=ds_weights)

    return test_ds
