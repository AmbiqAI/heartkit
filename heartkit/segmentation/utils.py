from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import tensorflow as tf
from rich.console import Console

from ..datasets import (
    HeartKitDataset,
    IcentiaDataset,
    LudbDataset,
    QtdbDataset,
    SyntheticDataset,
    augment_pipeline,
    preprocess_pipeline,
)
from ..defines import (
    HeartExportParams,
    HeartTask,
    HeartTestParams,
    HeartTrainParams,
    PreprocessParams,
)
from ..models import UNet, UNetBlockParams, UNetParams, generate_model

console = Console()


def prepare(x: npt.NDArray, sample_rate: float, preprocesses: list[PreprocessParams]) -> npt.NDArray:
    """Prepare dataset

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
    class_map: dict[int, int],
    dataset_names: list[str],
    num_pts: int = 200,
) -> list[HeartKitDataset]:
    """Load dataset

    Args:
        ds_path (Path): Path to dataset
        frame_size (int): Frame size
        sampling_rate (int): Sampling rate
        class_map (dict[int, int]): Class map
        dataset_names (list[str]): Dataset names
        num_pts (int, optional): Number of points. Defaults to 200.

    Returns:
        list[HeartKitDataset]: Dataset
    """
    datasets: list[HeartKitDataset] = []
    if "synthetic" in dataset_names:
        datasets.append(
            SyntheticDataset(
                ds_path,
                task=HeartTask.segmentation,
                frame_size=frame_size,
                target_rate=sampling_rate,
                class_map=class_map,
                num_pts=num_pts,
            )
        )
    if "ludb" in dataset_names:
        datasets.append(
            LudbDataset(
                ds_path,
                task=HeartTask.segmentation,
                frame_size=frame_size,
                target_rate=sampling_rate,
                class_map=class_map,
            )
        )
    if "qtdb" in dataset_names:
        datasets.append(
            QtdbDataset(
                ds_path,
                task=HeartTask.segmentation,
                frame_size=frame_size,
                target_rate=sampling_rate,
                class_map=class_map,
            )
        )
    if "icentia11k" in dataset_names:
        datasets.append(
            IcentiaDataset(
                ds_path,
                task=HeartTask.segmentation,
                frame_size=frame_size,
                target_rate=sampling_rate,
                class_map=class_map,
            )
        )
    return datasets


def load_train_datasets(
    datasets: list[HeartKitDataset],
    params: HeartTrainParams,
) -> tuple[tf.data.Dataset, tf.data.Dataset]:
    """Load segmentation train datasets.

    Args:
        datasets (list[HeartKitDataset]): Datasets
        params (HeartTrainParams): Train params

    Returns:
        tuple[tf.data.Dataset, tf.data.Dataset]: ds, train and validation datasets
    """

    def preprocess(x: npt.NDArray) -> npt.NDArray:
        xx = x.copy().squeeze()
        if params.augmentations:
            xx = augment_pipeline(xx, augmentations=params.augmentations, sample_rate=params.sampling_rate)
        xx = prepare(xx, sample_rate=params.sampling_rate, preprocesses=params.preprocesses)
        return xx

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
    ds_weights = np.array([len(ds.get_train_patient_ids()) for ds in datasets])
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
    datasets: list[HeartKitDataset],
    params: HeartTestParams | HeartExportParams,
) -> tuple[npt.NDArray, npt.NDArray]:
    """Load segmentation test dataset.

    Args:
        params (HeartTestParams|HeartExportParams): Test params

    Returns:
        tuple[npt.NDArray, npt.NDArray]: Test data and labels
    """

    def preprocess(x: npt.NDArray) -> npt.NDArray:
        xx = x.copy().squeeze()
        xx = prepare(xx, sample_rate=params.sampling_rate, preprocesses=params.preprocesses)
        return xx

    with console.status("[bold green] Loading test dataset..."):
        test_datasets = [
            ds.load_test_dataset(
                test_pt_samples=params.samples_per_patient,
                preprocess=preprocess,
                num_workers=params.data_parallelism,
            )
            for ds in datasets
        ]
        ds_weights = np.array([len(ds.get_test_patient_ids()) for ds in datasets])
        ds_weights = ds_weights / ds_weights.sum()

        test_ds = tf.data.Dataset.sample_from_datasets(test_datasets, weights=ds_weights)
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
    """Reference segmentation model

    Args:
        inputs (tf.Tensor): Model inputs
        num_classes (int): Number of classes

    Returns:
        tf.keras.Model: Model
    """
    blocks = [
        UNetBlockParams(filters=8, depth=2, ddepth=1, kernel=(1, 3), strides=(1, 2), skip=True),
        UNetBlockParams(filters=16, depth=2, ddepth=1, kernel=(1, 3), strides=(1, 2), skip=True),
        UNetBlockParams(filters=24, depth=2, ddepth=1, kernel=(1, 3), strides=(1, 2), skip=True),
        UNetBlockParams(filters=32, depth=2, ddepth=1, kernel=(1, 3), strides=(1, 2), skip=True),
        UNetBlockParams(filters=40, depth=2, ddepth=1, kernel=(1, 3), strides=(1, 2), skip=True),
    ]
    return UNet(
        inputs,
        params=UNetParams(
            blocks=blocks,
            output_kernel_size=(1, 3),
            include_top=True,
        ),
        num_classes=num_classes,
    )
