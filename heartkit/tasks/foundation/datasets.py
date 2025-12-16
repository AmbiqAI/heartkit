import tensorflow as tf
import helia_edge as helia

from ...datasets import HKDataset, create_augmentation_pipeline
from ..utils import load_train_dataloader_split, load_test_dataloader_split
from ...defines import HKTaskParams, NamedParams

from .dataloaders import FoundationTaskFactory

logger = helia.utils.setup_logger(__name__)


def create_data_pipeline(
    ds: tf.data.Dataset,
    sampling_rate: int,
    batch_size: int,
    buffer_size: int | None = None,
    preprocesses: list[NamedParams] | None = None,
    augmentations: list[NamedParams] | None = None,
) -> tf.data.Dataset:
    """Create a beat task data pipeline for given dataset.

    Args:
        ds (tf.data.Dataset): Input dataset.
        sampling_rate (int): Sampling rate of the dataset.
        batch_size (int): Batch size.
        buffer_size (int, optional): Buffer size for shuffling. Defaults to None.
        preprocesses (list[NamedParams], optional): List of preprocesses. Defaults to None.
        augmentations (list[NamedParams], optional): List of augmentations. Defaults to None.

    Returns:
        tf.data.Dataset: Data pipeline.
    """
    augmenter = create_augmentation_pipeline(augmentations + preprocesses, sampling_rate)
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
    ds = ds.map(
        lambda x1, x2: {
            helia.trainers.SimCLRTrainer.SAMPLES: x1,
            helia.trainers.SimCLRTrainer.AUG_SAMPLES_0: augmenter(x1),
            helia.trainers.SimCLRTrainer.AUG_SAMPLES_1: augmenter(x2),
        },
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    return ds.prefetch(tf.data.AUTOTUNE)


def load_train_datasets(
    datasets: list[HKDataset],
    params: HKTaskParams,
) -> tuple[tf.data.Dataset, tf.data.Dataset]:
    """ "Create 'tf.data.Dataset' pipeline.

    Args:
        datasets (list[HKDataset]): List of datasets.
        params (HKTaskParams): Task parameters.

    Returns:
        tf.data.Dataset: Augmented dataset
    """

    train_ds, val_ds = load_train_dataloader_split(datasets, params, factory=FoundationTaskFactory)

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
        preprocesses=params.preprocesses,
        augmentations=params.augmentations,
    )

    # If given fixed val size or steps, then capture and cache
    val_steps_per_epoch = params.val_size // params.batch_size if params.val_size else params.val_steps_per_epoch
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
    test_ds = load_test_dataloader_split(datasets, params, factory=FoundationTaskFactory)

    test_ds = create_data_pipeline(
        ds=test_ds,
        sampling_rate=params.sampling_rate,
        batch_size=params.batch_size,
        preprocesses=params.preprocesses,
        augmentations=params.augmentations,
    )

    batch_size = getattr(params, "batch_size", 1)
    test_ds = test_ds.take(params.test_size // batch_size).cache()

    return test_ds
