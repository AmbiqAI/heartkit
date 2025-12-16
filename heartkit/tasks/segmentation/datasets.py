import tensorflow as tf
import helia_edge as helia

from ...datasets import HKDataset, create_augmentation_pipeline
from ...defines import HKTaskParams, NamedParams
from ..utils import load_train_dataloader_split, load_test_dataloader_split
from .dataloaders import SegmentationDataloaderFactory as DataloaderFactory

logger = helia.utils.setup_logger(__name__)


def create_data_pipeline(
    ds: tf.data.Dataset,
    sampling_rate: int,
    batch_size: int,
    buffer_size: int | None = None,
    augmentations: list[NamedParams] | None = None,
    num_classes: int = 2,
) -> tf.data.Dataset:
    """Create data pipeline for training

    Args:
        ds (tf.data.Dataset): Dataset
        sampling_rate (int): Sampling rate
        batch_size (int): Batch size
        buffer_size (int, optional): Buffer size. Defaults to None.
        augmentations (list[NamedParams], optional): Augmentations. Defaults to None.
        num_classes (int, optional): Number of classes. Defaults to 2.

    Returns:
        tf.data.Dataset: Data pipeline
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
    augmenter = create_augmentation_pipeline(
        augmentations,
        sampling_rate=sampling_rate,
    )
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
    """Load training and validation datasets

    Args:
        datasets (list[HKDataset]): List of datasets
        params (HKTaskParams): Task parameters

    Returns:
        tuple[tf.data.Dataset, tf.data.Dataset]: Training and validation datasets
    """

    train_ds, val_ds = load_train_dataloader_split(datasets, params, factory=DataloaderFactory)

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
        buffer_size=None,
        augmentations=params.augmentations + params.preprocesses,
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
    """Load test dataset

    Args:
        datasets (list[HKDataset]): List of datasets
        params (HKTaskParams): Task parameters

    Returns:
        tf.data.Dataset: Test dataset
    """

    test_ds = load_test_dataloader_split(datasets, params, factory=DataloaderFactory)

    test_ds = create_data_pipeline(
        ds=test_ds,
        sampling_rate=params.sampling_rate,
        batch_size=params.batch_size,
        augmentations=params.augmentations + params.preprocesses,
        num_classes=params.num_classes,
    )

    if params.test_size:
        batch_size = getattr(params, "batch_size", 1)
        test_ds = test_ds.take(params.test_size // batch_size).cache()

    return test_ds
