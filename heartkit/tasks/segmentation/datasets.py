import functools
import logging
from pathlib import Path

import numpy as np
import numpy.typing as npt
import tensorflow as tf

from ...datasets import (
    HKDataset,
    augment_pipeline,
    preprocess_pipeline,
    uniform_id_generator,
)
from ...datasets.dataloader import test_dataloader, train_val_dataloader
from ...defines import (
    AugmentationParams,
    HKExportParams,
    HKTestParams,
    HKTrainParams,
    PreprocessParams,
)
from ...utils import resolve_template_path
from .dataloaders import (
    icentia11k_data_generator,
    icentia11k_label_map,
    ludb_data_generator,
    ludb_label_map,
    ptbxl_data_generator,
    ptbxl_label_map,
    synthetic_data_generator,
    synthetic_label_map,
)

logger = logging.getLogger(__name__)


def preprocess(x: npt.NDArray, preprocesses: list[PreprocessParams], sample_rate: float) -> npt.NDArray:
    """Preprocess data pipeline

    Args:
        x (npt.NDArray): Input data
        preprocesses (list[PreprocessParams]): Preprocess parameters
        sample_rate (float): Sample rate

    Returns:
        npt.NDArray: Preprocessed data
    """
    return preprocess_pipeline(x, preprocesses=preprocesses, sample_rate=sample_rate)


def augment(x: npt.NDArray, augmentations: list[AugmentationParams], sample_rate: float) -> npt.NDArray:
    """Augment data pipeline

    Args:
        x (npt.NDArray): Input data
        augmentations (list[AugmentationParams]): Augmentation parameters
        sample_rate (float): Sample rate

    Returns:
        npt.NDArray: Augmented data
    """
    return augment_pipeline(
        x=x,
        augmentations=augmentations,
        sample_rate=sample_rate,
    )


def prepare(
    x_y: tuple[npt.NDArray, npt.NDArray],
    sample_rate: float,
    preprocesses: list[PreprocessParams],
    augmentations: list[AugmentationParams],
    spec: tuple[tf.TensorSpec, tf.TensorSpec],
    num_classes: int,
) -> tuple[npt.NDArray, npt.NDArray]:
    """Prepare dataset

    Args:
        x_y (tuple[npt.NDArray, int]): Input data and label
        sample_rate (float): Sample rate
        preprocesses (list[PreprocessParams]|None): Preprocess parameters
        augmentations (list[AugmentationParams]|None): Augmentation parameters
        spec (tuple[tf.TensorSpec, tf.TensorSpec]): TensorSpec
        num_classes (int): Number of classes

    Returns:
        tuple[npt.NDArray, npt.NDArray]: Data and label
    """
    x, y = x_y[0].copy(), x_y[1]

    if augmentations:
        x = augment(x, augmentations, sample_rate)
    # END IF

    if preprocesses:
        x = preprocess(x, preprocesses, sample_rate)
    # END IF

    x = x.reshape(spec[0].shape)
    y = tf.one_hot(y, num_classes)

    return x, y


def get_ds_label_map(ds: HKDataset, label_map: dict[int, int] | None = None) -> dict[int, int]:
    """Get label map for dataset

    Args:
        ds (HKDataset): Dataset
        label_map (dict[int, int]|None): Label map

    Returns:
        dict[int, int]: Label map
    """
    match ds.name:
        case "icentia11k":
            return icentia11k_label_map(label_map=label_map)
        case "ludb":
            return ludb_label_map(label_map=label_map)
        case "ptbxl":
            return ptbxl_label_map(label_map=label_map)
        case "synthetic":
            return synthetic_label_map(label_map=label_map)
        case _:
            raise ValueError(f"Dataset {ds.name} not supported")
    # END MATCH


def get_data_generator(
    ds: HKDataset, frame_size: int, samples_per_patient: int, target_rate: int, label_map: dict[int, int] | None = None
):
    """Get task data generator for dataset

    Args:
        ds (HKDataset): Dataset
        frame_size (int): Frame size
        samples_per_patient (int): Samples per patient
        target_rate (int): Target rate
        label_map (dict[int, int]|None): Label map

    Returns:
        callable: Data generator
    """
    match ds.name:
        case "icentia11k":
            data_generator = icentia11k_data_generator
        case "ludb":
            data_generator = ludb_data_generator
        case "ptbxl":
            data_generator = ptbxl_data_generator
        case "synthetic":
            data_generator = synthetic_data_generator

        case _:
            raise ValueError(f"Dataset {ds.name} not supported")
    # END MATCH
    return functools.partial(
        data_generator,
        ds=ds,
        frame_size=frame_size,
        samples_per_patient=samples_per_patient,
        target_rate=target_rate,
        label_map=label_map,
    )


def resolve_ds_cache_path(fpath: Path | None, ds: HKDataset, task: str, frame_size: int, sample_rate: int):
    """Resolve dataset cache path

    Args:
        fpath (Path|None): File path
        ds (HKDataset): Dataset
        task (str): Task
        frame_size (int): Frame size
        sample_rate (int): Sampling rate

    Returns:
        Path|None: Resolved path
    """
    if not fpath:
        return None
    return resolve_template_path(
        fpath=fpath,
        dataset=ds.name,
        task=task,
        frame_size=frame_size,
        sampling_rate=sample_rate,
    )


def load_train_datasets(
    datasets: list[HKDataset],
    params: HKTrainParams,
    ds_spec: tuple[tf.TensorSpec, tf.TensorSpec],
) -> tuple[tf.data.Dataset, tf.data.Dataset]:
    """Load training and validation datasets

    Args:
        datasets (list[HKDataset]): Datasets
        params (HKTrainParams): Training parameters
        ds_spec (tuple[tf.TensorSpec, tf.TensorSpec]): TensorSpec

    Returns:
        tuple[tf.data.Dataset, tf.data.Dataset]: Train and validation datasets
    """
    id_generator = functools.partial(uniform_id_generator, repeat=True)
    train_prepare = functools.partial(
        prepare,
        sample_rate=params.sampling_rate,
        preprocesses=params.preprocesses,
        augmentations=params.augmentations,
        spec=ds_spec,
        num_classes=params.num_classes,
    )

    train_datasets = []
    val_datasets = []
    for ds in datasets:

        val_file = resolve_ds_cache_path(
            params.val_file, ds=ds, task="segmentation", frame_size=params.frame_size, sample_rate=params.sampling_rate
        )
        data_generator = get_data_generator(
            ds=ds,
            frame_size=params.frame_size,
            samples_per_patient=params.samples_per_patient,
            target_rate=params.sampling_rate,
            label_map=params.class_map,
        )
        train_ds, val_ds = train_val_dataloader(
            ds=ds,
            spec=ds_spec,
            data_generator=data_generator,
            id_generator=id_generator,
            train_patients=params.train_patients,
            val_patients=params.val_patients,
            val_pt_samples=params.val_samples_per_patient,
            val_file=val_file,
            val_size=params.val_size,
            label_map=params.class_map,
            label_type=None,
            preprocess=train_prepare,
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


def load_test_dataset(
    datasets: list[HKDataset],
    params: HKTestParams | HKExportParams,
    ds_spec: tuple[tf.TensorSpec, tf.TensorSpec],
) -> tf.data.Dataset:
    """Load test dataset

    Args:
        datasets (list[HKDataset]): Datasets
        params (HKTestParams|HKExportParams): Test parameters
        ds_spec (tuple[tf.TensorSpec, tf.TensorSpec]): TensorSpec

    Returns:
        tf.data.Dataset: Test dataset
    """

    id_generator = functools.partial(uniform_id_generator, repeat=True)
    test_prepare = functools.partial(
        prepare,
        sample_rate=params.sampling_rate,
        preprocesses=params.preprocesses,
        augmentations=None,  # params.augmentations,
        spec=ds_spec,
        num_classes=params.num_classes,
    )
    test_datasets = []
    for ds in datasets:

        test_file = resolve_ds_cache_path(
            fpath=params.test_file,
            ds=ds,
            task="segmentation",
            frame_size=params.frame_size,
            sample_rate=params.sampling_rate,
        )
        data_generator = get_data_generator(
            ds=ds,
            frame_size=params.frame_size,
            samples_per_patient=params.test_samples_per_patient,
            target_rate=params.sampling_rate,
            label_map=params.class_map,
        )
        test_ds = test_dataloader(
            ds=ds,
            spec=ds_spec,
            data_generator=data_generator,
            id_generator=id_generator,
            test_patients=params.test_patients,
            test_file=test_file,
            label_map=params.class_map,
            label_type=None,
            preprocess=test_prepare,
            num_workers=params.data_parallelism,
        )
        test_datasets.append(test_ds)
    # END FOR

    ds_weights = np.array([d.weight for d in params.datasets])
    ds_weights = ds_weights / ds_weights.sum()

    test_ds = tf.data.Dataset.sample_from_datasets(test_datasets, weights=ds_weights)

    # END WITH
    return test_ds
