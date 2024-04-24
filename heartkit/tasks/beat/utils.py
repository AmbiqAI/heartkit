from pathlib import Path

import keras
import numpy as np
import numpy.typing as npt
import tensorflow as tf
from rich.console import Console

from ...datasets import DatasetFactory, HKDataset, augment_pipeline, preprocess_pipeline
from ...defines import (
    DatasetParams,
    HKExportParams,
    HKTestParams,
    HKTrainParams,
    ModelArchitecture,
    PreprocessParams,
)
from ...models import EfficientNetParams, EfficientNetV2, MBConvParams, ModelFactory

console = Console()


def get_feat_shape(frame_size: int) -> tuple[int, ...]:
    """Get dataset feature shape.

    Args:
        frame_size (int): Frame size

    Returns:
        tuple[int, ...]: Feature shape
    """
    return (frame_size, 1)  # Time x Channels


def get_class_shape(frame_size: int, nclasses: int) -> tuple[int, ...]:
    """Get dataset class shape.

    Args:
        frame_size (int): Frame size
        nclasses (int): Number of classes

    Returns:
        tuple[int, ...]: Class shape

    """
    return (nclasses,)  # One-hot encoded classes


def prepare(x: npt.NDArray, sample_rate: float, preprocesses: list[PreprocessParams]) -> npt.NDArray:
    """Prepare dataset.

    Args:
        x (npt.NDArray): Input signal
        sample_rate (float): Sampling rate
        preprocesses (list[PreprocessParams]): Preprocessing pipeline

    Returns:
        npt.NDArray: Prepared signal
    """
    if not preprocesses:
        preprocesses = [
            dict(name="filter", args=dict(axis=0, lowcut=0.5, highcut=30, order=3, sample_rate=sample_rate)),
            dict(name="znorm", args=dict(axis=None, eps=0.1)),
        ]
    return preprocess_pipeline(x, preprocesses=preprocesses, sample_rate=sample_rate)


def load_datasets(
    ds_path: Path,
    frame_size: int,
    sampling_rate: int,
    class_map: dict[int, int] | None,
    spec: tuple[tf.TensorSpec, tf.TensorSpec],
    datasets: list[DatasetParams] = None,
) -> list[HKDataset]:
    """Load datasets

    Args:
        ds_path (Path): Path to dataset
        frame_size (int): Frame size
        sampling_rate (int): Sampling rate
        spec (tuple[tf.TensorSpec, tf.TensorSpec]): feat/class shape specs
        class_map (dict[int, int]): Class map
        datasets (list[DatasetParams]): List of datasets

    Returns:
        HeartKitDataset: Dataset
    """
    dsets = []
    for dset in datasets:
        if DatasetFactory.has(dset.name):
            dsets.append(
                DatasetFactory.get(dset.name)(
                    ds_path=ds_path,
                    task="beat",
                    frame_size=frame_size,
                    target_rate=sampling_rate,
                    class_map=class_map,
                    spec=spec,
                    **dset.params
                )
            )
        # END IF
    # END FOR
    return dsets


def load_train_datasets(
    datasets: list[HKDataset],
    params: HKTrainParams,
) -> tuple[tf.data.Dataset, tf.data.Dataset]:
    """Load beat train datasets.

    Args:
        datasets (list[HeartKitDataset]): Datasets
        params (HKTrainParams): Train params

    Returns:
        tuple[tf.data.Dataset, tf.data.Dataset]: Train and validation datasets
    """

    feat_shape = get_feat_shape(params.frame_size)

    def preprocess(x_y: tuple[npt.NDArray, npt.NDArray]) -> tuple[npt.NDArray, npt.NDArray]:
        xx = x_y[0].copy()
        yy = x_y[1]
        # Augment each channel
        if params.augmentations:
            xx = augment_pipeline(
                x=xx,
                augmentations=params.augmentations,
                sample_rate=params.sampling_rate,
            )
        # END IF
        xx = prepare(xx, sample_rate=params.sampling_rate, preprocesses=params.preprocesses).reshape(feat_shape)
        yy = tf.one_hot(yy, params.num_classes)
        return xx, yy

    train_datasets = []
    val_datasets = []
    for ds in datasets:
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
        train_datasets.append(train_ds)
        val_datasets.append(val_ds)
    # END FOR

    ds_weights = np.array([d.weight for d in params.datasets])
    ds_weights = ds_weights / ds_weights.sum()

    train_ds = tf.data.Dataset.sample_from_datasets(train_datasets, weights=ds_weights)
    val_ds = tf.data.Dataset.sample_from_datasets(val_datasets, weights=ds_weights)

    # Shuffle and batch datasets for training
    train_ds = (
        train_ds.shuffle(
            buffer_size=params.buffer_size,
            reshuffle_each_iteration=True,
        )
        .batch(
            batch_size=params.batch_size,
            drop_remainder=False,
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )
    val_ds = val_ds.batch(
        batch_size=params.batch_size,
        drop_remainder=True,
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    return train_ds, val_ds


def load_test_datasets(
    datasets: list[HKDataset],
    params: HKTestParams | HKExportParams,
) -> tuple[HKDataset, npt.NDArray, npt.NDArray]:
    """Load beat test dataset.

    Args:
        datasets (list[HeartKitDataset]): Datasets
        params (HKTestParams|HKExportParams): Test params

    Returns:
        tuple[npt.NDArray, npt.NDArray]: Test data and labels
    """

    feat_shape = get_feat_shape(params.frame_size)

    def preprocess(x_y: tuple[npt.NDArray, npt.NDArray]) -> tuple[npt.NDArray, npt.NDArray]:
        xx = x_y[0].copy().squeeze()
        xx = prepare(xx, sample_rate=params.sampling_rate, preprocesses=params.preprocesses).reshape(feat_shape)
        yy = tf.one_hot(x_y[1], params.num_classes)
        return xx, yy

    with console.status("[bold green] Loading test dataset..."):
        test_datasets = [
            ds.load_test_dataset(
                test_pt_samples=params.test_samples_per_patient,
                test_file=params.test_file,
                preprocess=preprocess,
                num_workers=params.data_parallelism,
            )
            for ds in datasets
        ]

        ds_weights = np.array([d.weight for d in params.datasets])
        ds_weights = ds_weights / ds_weights.sum()

        test_ds = tf.data.Dataset.sample_from_datasets(test_datasets, weights=ds_weights)
        test_x, test_y = next(test_ds.batch(params.test_size).as_numpy_iterator())
    # END WITH
    return test_x, test_y


def create_model(inputs: tf.Tensor, num_classes: int, architecture: ModelArchitecture | None) -> keras.Model:
    """Generate model or use default

    Args:
        inputs (tf.Tensor): Model inputs
        num_classes (int): Number of classes
        architecture (ModelArchitecture|None): Model

    Returns:
        keras.Model: Model
    """
    if architecture:
        return ModelFactory.create(
            name=architecture.name,
            params=architecture.params,
            inputs=inputs,
            num_classes=num_classes,
        )

    return _default_model(inputs=inputs, num_classes=num_classes)


def _default_model(
    inputs: tf.Tensor,
    num_classes: int,
) -> keras.Model:
    """Reference beat model

    Args:
        inputs (tf.Tensor): Model inputs
        num_classes (int): Number of classes

    Returns:
        keras.Model: Model
    """
    blocks = [
        MBConvParams(
            filters=32,
            depth=2,
            ex_ratio=1,
            kernel_size=(1, 3),
            strides=(1, 1),
            se_ratio=2,
        ),
        MBConvParams(
            filters=48,
            depth=2,
            ex_ratio=1,
            kernel_size=(1, 3),
            strides=(1, 2),
            se_ratio=2,
        ),
        MBConvParams(
            filters=64,
            depth=3,
            ex_ratio=1,
            kernel_size=(1, 3),
            strides=(1, 2),
            se_ratio=4,
        ),
        MBConvParams(
            filters=96,
            depth=3,
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
            input_strides=(1, 2),
            input_kernel_size=(1, 5),
            output_filters=0,
            blocks=blocks,
            include_top=True,
            dropout=0.0,
            drop_connect_rate=0.0,
        ),
        num_classes=num_classes,
    )
