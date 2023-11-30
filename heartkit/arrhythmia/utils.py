from pathlib import Path
from typing import Any, cast

import numpy.typing as npt
import physiokit as pk
import tensorflow as tf
from rich.console import Console

from ..datasets import HeartKitDataset, IcentiaDataset, augment_pipeline
from ..defines import HeartExportParams, HeartTask, HeartTestParams, HeartTrainParams
from ..models import EfficientNetParams, EfficientNetV2, MBConvParams, generate_model

console = Console()


def prepare(x: npt.NDArray, sample_rate: float) -> npt.NDArray:
    """Prepare dataset."""
    x = pk.signal.filter_signal(
        x,
        lowcut=0.5,
        highcut=30,
        order=3,
        sample_rate=sample_rate,
        axis=0,
        forward_backward=True,
    )
    x = pk.signal.normalize_signal(x, eps=0.1, axis=None)
    return x


def load_dataset(ds_path: Path, frame_size: int, sampling_rate: int, class_map: dict[int, int]) -> HeartKitDataset:
    """Load dataset

    Args:
        ds_path (Path): Path to dataset
        frame_size (int): Frame size
        sampling_rate (int): Sampling rate
        class_map (dict[int, int]): Class map

    Returns:
        HeartKitDataset: Dataset
    """
    ds = IcentiaDataset(
        ds_path=ds_path,
        task=HeartTask.arrhythmia,
        frame_size=frame_size,
        target_rate=sampling_rate,
        class_map=class_map,
    )
    return ds


def load_train_datasets(
    ds: HeartKitDataset,
    params: HeartTrainParams,
) -> tuple[tf.data.Dataset, tf.data.Dataset]:
    """Load arrhythmia train datasets.

    Args:
        ds (HeartKitDataset): Dataset
        params (HeartTrainParams): Train params

    Returns:
        tuple[tf.data.Dataset, tf.data.Dataset]: Train and validation datasets
    """

    def preprocess(x: npt.NDArray) -> npt.NDArray:
        xx = x.copy().squeeze()
        if params.augmentations:
            xx = augment_pipeline(xx, augmentations=params.augmentations, sample_rate=params.sampling_rate)
        xx = prepare(xx, sample_rate=params.sampling_rate)
        return xx

    # Create TF datasets
    train_ds, val_ds = ds.load_train_datasets(
        train_patients=params.train_patients,
        val_patients=params.val_patients,
        train_pt_samples=params.samples_per_patient,
        val_pt_samples=params.val_samples_per_patient,
        val_file=params.val_file,
        val_size=params.val_size,
        preprocess=preprocess,
        num_workers=params.data_parallelism,
    )

    # Shuffle and batch datasets for training
    train_ds = cast(
        tf.data.Dataset,
        train_ds.shuffle(
            buffer_size=params.buffer_size,
            reshuffle_each_iteration=True,
        )
        .batch(
            batch_size=params.batch_size,
            drop_remainder=True,
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        .prefetch(buffer_size=tf.data.AUTOTUNE),
    )
    val_ds = cast(
        tf.data.Dataset,
        val_ds.batch(
            batch_size=params.batch_size,
            drop_remainder=True,
            num_parallel_calls=tf.data.AUTOTUNE,
        ),
    )
    return train_ds, val_ds


def load_test_dataset(
    ds: HeartKitDataset,
    params: HeartTestParams | HeartExportParams,
) -> tuple[npt.NDArray, npt.NDArray]:
    """Load arrhythmia test dataset.

    Args:
        ds (HeartKitDataset): Dataset
        params (HeartTestParams|HeartExportParams): Test params

    Returns:
        tuple[npt.NDArray, npt.NDArray]: Test data and labels
    """

    def preprocess(x: npt.NDArray) -> npt.NDArray:
        xx = x.copy().squeeze()
        xx = prepare(xx, sample_rate=params.sampling_rate)
        return xx

    with console.status("[bold green] Loading test dataset..."):
        test_ds = ds.load_test_dataset(
            test_patients=params.test_patients,
            test_pt_samples=params.samples_per_patient,
            preprocess=preprocess,
            num_workers=params.data_parallelism,
        )
        test_x, test_y = next(test_ds.batch(params.test_size).as_numpy_iterator())
    # END WITH
    return test_x, test_y


def create_model(
    inputs: tf.Tensor, num_classes: int, name: str | None = None, params: dict[str, Any] | None = None
) -> tf.keras.Model:
    """Generate model or use default

    Args:
        inputs (tf.Tensor): Model inputs
        num_classes (int): Number of classes
        name (str | None, optional): Architecture type. Defaults to None.
        params (dict[str, Any] | None, optional): Model parameters. Defaults to None.

    Returns:
        tf.keras.Model: Model
    """
    if name:
        return generate_model(inputs=inputs, num_classes=num_classes, name=name, params=params)

    return _default_model(inputs=inputs, num_classes=num_classes)


def _default_model(
    inputs: tf.Tensor,
    num_classes: int,
) -> tf.keras.Model:
    """Reference arrhythmia model

    Args:
        inputs (tf.Tensor): Model inputs
        num_classes (int): Number of classes

    Returns:
        tf.keras.Model: Model
    """

    blocks = [
        MBConvParams(
            filters=32,
            depth=2,
            ex_ratio=1,
            kernel_size=(1, 3),
            strides=(1, 2),
            se_ratio=2,
        ),
        MBConvParams(
            filters=48,
            depth=1,
            ex_ratio=1,
            kernel_size=(1, 3),
            strides=(1, 2),
            se_ratio=4,
        ),
        MBConvParams(
            filters=64,
            depth=2,
            ex_ratio=1,
            kernel_size=(1, 3),
            strides=(1, 2),
            se_ratio=4,
        ),
        MBConvParams(
            filters=80,
            depth=1,
            ex_ratio=1,
            kernel_size=(1, 3),
            strides=(1, 2),
            se_ratio=4,
        ),
    ]
    return EfficientNetV2(
        inputs,
        params=EfficientNetParams(
            input_filters=24,
            input_kernel_size=(1, 3),
            input_strides=(1, 2),
            blocks=blocks,
            output_filters=0,
            include_top=True,
            dropout=0.0,
            drop_connect_rate=0.0,
        ),
        num_classes=num_classes,
    )
