import numpy as np
import tensorflow as tf
import neuralspot_edge as nse

from ...datasets import HKDataset, create_augmentation_pipeline
from ...defines import HKTaskParams, NamedParams
from .dataloader import DenoiseDataloader

logger = nse.utils.setup_logger(__name__)


def create_data_pipeline(
    ds: tf.data.Dataset,
    sampling_rate: int,
    batch_size: int,
    buffer_size: int | None = None,
    preprocesses: list[NamedParams] | None = None,
    augmentations: list[NamedParams] | None = None,
) -> tf.data.Dataset:
    """ "Create 'tf.data.Dataset' pipeline.

    Args:
        ds (tf.data.Dataset): Input dataset
        sampling_rate (int): Sampling rate
        batch_size (int): Batch size
        buffer_size (int | None, optional): Buffer size. Defaults to None.
        preprocesses (list[NamedParams] | None, optional): Preprocessing pipeline. Defaults to None.
        augmentations (list[NamedParams] | None, optional): Augmentation pipeline. Defaults to None.

    Returns:
        tf.data.Dataset: Augmented dataset
    """
    preprocessor = create_augmentation_pipeline(preprocesses, sampling_rate)
    augmenter = create_augmentation_pipeline(augmentations, sampling_rate)
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
    ds = ds.map(lambda x: preprocessor(x), num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.map(lambda x: (augmenter(x), x), num_parallel_calls=tf.data.AUTOTUNE)
    return ds.prefetch(tf.data.AUTOTUNE)


def load_train_datasets(
    datasets: list[HKDataset],
    params: HKTaskParams,
) -> tuple[tf.data.Dataset, tf.data.Dataset]:
    """Load training and validation dataset pipelines

    Args:
        datasets (list[HKDataset]): List of datasets
        params (HKTaskParams): Training parameters
    """
    train_datasets = []
    val_datasets = []
    for ds in datasets:
        dataloader = DenoiseDataloader(
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
        preprocesses=params.preprocesses,
        augmentations=params.augmentations,
    )

    val_ds = create_data_pipeline(
        ds=val_ds,
        sampling_rate=params.sampling_rate,
        batch_size=params.batch_size,
        buffer_size=None,
        preprocesses=params.preprocesses,
        augmentations=params.augmentations,
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
    """Load test dataset pipeline

    Args:
        datasets (list[HKDataset]): List of datasets
        params (HKTaskParams): Test or export parameters

    Returns:
        tf.data.Dataset: Test dataset pipeline
    """
    test_datasets = []
    for ds in datasets:
        dataloader = DenoiseDataloader(
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
        buffer_size=None,
        preprocesses=params.preprocesses,
        augmentations=params.augmentations,
    )

    if params.test_size:
        batch_size = getattr(params, "batch_size", 1)
        test_ds = test_ds.take(params.test_size // batch_size).cache()

    return test_ds
