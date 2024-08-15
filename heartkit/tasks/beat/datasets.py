import numpy as np
import tensorflow as tf
import neuralspot_edge as nse

from ...datasets import (
    HKDataset,
    create_augmentation_pipeline,
)
from ...datasets.dataloader import HKDataloader
from ...defines import HKTaskParams, NamedParams

from .dataloaders import BeatTaskFactory

logger = nse.utils.setup_logger(__name__)


def create_data_pipeline(
    ds: tf.data.Dataset,
    sampling_rate: int,
    batch_size: int,
    buffer_size: int | None = None,
    augmentations: list[NamedParams] | None = None,
    num_classes: int = 2,
) -> tf.data.Dataset:
    """Create a beat task data pipeline for given dataset.

    Args:
        ds (tf.data.Dataset): Input dataset.
        sampling_rate (int): Sampling rate of the dataset.
        batch_size (int): Batch size.
        buffer_size (int, optional): Buffer size for shuffling. Defaults to None.
        augmentations (list[NamedParams], optional): List of augmentations. Defaults to None.
        num_classes (int, optional): Number of classes. Defaults to 2.

    Returns:
        tf.data.Dataset: Data pipeline.
    """
    if buffer_size:
        ds = ds.shuffle(
            buffer_size=buffer_size,
            reshuffle_each_iteration=True,
        )
    if batch_size:
        ds = ds.batch(
            batch_size=batch_size,
            drop_remainder=True,
            num_parallel_calls=tf.data.AUTOTUNE,
        )
    augmenter = create_augmentation_pipeline(augmentations, sampling_rate=sampling_rate)
    ds = (
        ds.map(
            lambda data, labels: {
                "data": tf.cast(data, "float32"),
                "labels": tf.one_hot(labels, num_classes),
            },
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        .map(
            augmenter,
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        .map(
            lambda data: (data["data"], data["labels"]),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
    )

    return ds.prefetch(tf.data.AUTOTUNE)


def load_train_datasets(
    datasets: list[HKDataset],
    params: HKTaskParams,
) -> tuple[tf.data.Dataset, tf.data.Dataset]:
    """Load training and validation tf.data.Datasets pipeline.

    !!! note
        if val_size or val_steps_per_epoch is given, then validation dataset will be
        a fixed cached size. Otherwise, it will be a unbounded dataset generator. In
        the latter case, a length will need to be passed to functions like `model.fit`.

    Args:
        datasets (list[HKDataset]): List of datasets.
        params (HKTaskParams): Training parameters.

    Returns:
        tuple[tf.data.Dataset, tf.data.Dataset]: Training and validation datasets
    """
    train_datasets = []
    val_datasets = []
    for ds in datasets:
        dataloader: HKDataloader = BeatTaskFactory.get(ds.name)(
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

    # Shuffle and batch datasets for training
    train_ds = create_data_pipeline(
        ds=train_ds,
        sampling_rate=params.sampling_rate,
        batch_size=params.batch_size,
        buffer_size=params.buffer_size,
        augmentations=params.augmentations + params.preprocesses,
        num_classes=params.num_classes,
    )

    val_ds = create_data_pipeline(
        ds=val_ds,
        sampling_rate=params.sampling_rate,
        batch_size=params.batch_size,
        augmentations=params.preprocesses,
        num_classes=params.num_classes,
    )

    # If given fixed val size or steps, then capture and cache
    val_steps_per_epoch = params.val_size // params.batch_size if params.val_size else params.val_steps_per_epoch
    if val_steps_per_epoch:
        logger.info(f"Validation steps per epoch: {val_steps_per_epoch}")
        val_ds = val_ds.take(val_steps_per_epoch).cache()

    return train_ds, val_ds


def load_test_dataset(
    datasets: list[HKDataset],
    params: HKTaskParams,
) -> tf.data.Dataset:
    """Load test tf.data.Dataset.

    Args:
        datasets (list[HKDataset]): List of datasets.
        params (HKTaskParams): Test parameters.

    Returns:
        tf.data.Dataset: Test dataset
    """
    test_datasets = []
    for ds in datasets:
        dataloader: HKDataloader = BeatTaskFactory.get(ds.name)(
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
    test_ds = create_data_pipeline(
        ds=test_ds,
        sampling_rate=params.sampling_rate,
        batch_size=params.batch_size,
        augmentations=params.preprocesses,
        num_classes=params.num_classes,
    )

    if params.test_size:
        batch_size = getattr(params, "batch_size", 1)
        test_ds = test_ds.take(params.test_size // batch_size).cache()

    return test_ds
