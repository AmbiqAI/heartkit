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
from ...models import ModelFactory, UNet, UNetBlockParams, UNetParams
from .defines import HeartSegment

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
    return (frame_size, nclasses)  # Time x Classes


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
                    task="segmentation",
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


def apply_augmentation_pipeline(
    x: npt.NDArray,
    y: npt.NDArray,
    frame_size: int,
    sample_rate: int,
    class_map: dict[int, int],
    augmentations: list[PreprocessParams],
) -> tuple[npt.NDArray, npt.NDArray]:
    """Task augmentation pipeline

    Args:
        x (npt.NDArray): Input signal
        y (npt.NDArray): Target signal
        augmentations (list[PreprocessParams]): Augmentations

    Returns:
        tuple[npt.NDArray, npt.NDArray]: Augmented input and target signals
    """
    feat_shape = get_feat_shape(frame_size)

    x_mu, x_sd = np.nanmean(x), np.nanstd(x)
    # Standard augmentations dont impact the label
    x = augment_pipeline(x, augmentations=augmentations, sample_rate=sample_rate)

    # Augmentations that impact the label
    noise_label = class_map.get(HeartSegment.noise, None)
    if noise_label:

        cutout = next(filter(lambda a: a.name == "cutout", augmentations), None)
        if cutout:
            prob = cutout.params.get("probability", [0, 0.25])[1]
            width = cutout.params.get("width", [0, 1])
            if np.random.rand() < prob:
                dur = int(np.random.uniform(width[0], width[1]) * feat_shape[0])
                start = np.random.randint(0, feat_shape[0] - dur)
                stop = start + dur
                x[start:stop] = x_mu
                y[start:stop] = noise_label
            # END IF
        # END IF

        whiteout = next(filter(lambda a: a.name == "whiteout", augmentations), None)
        if whiteout:
            prob = whiteout.params.get("probability", [0, 0.25])[1]
            amp = whiteout.params.get("amplitude", [0.5, 1.0])
            width = whiteout.params.get("width", [0, 1])
            if np.random.rand() < prob:
                dur = int(np.random.uniform(width[0], width[1]) * feat_shape[0])
                start = np.random.randint(0, feat_shape[0] - dur)
                stop = start + dur
                scale = np.random.uniform(amp[0], amp[1]) * x_sd
                x[start:stop] += np.random.normal(0, scale, size=x[start:stop].shape)
                y[start:stop] = noise_label
            # END IF
        # END IF
    # END IF

    return x, y


def load_train_datasets(
    datasets: list[HKDataset],
    params: HKTrainParams,
) -> tuple[tf.data.Dataset, tf.data.Dataset]:
    """Load segmentation train datasets.

    Args:
        datasets (list[HeartKitDataset]): Datasets
        params (HKTrainParams): Train params

    Returns:
        tuple[tf.data.Dataset, tf.data.Dataset]: ds, train and validation datasets
    """

    feat_shape = get_feat_shape(params.frame_size)

    def preprocess(x_y: tuple[npt.NDArray, npt.NDArray]) -> tuple[npt.NDArray, npt.NDArray]:
        xx = x_y[0].copy().squeeze()
        yy = x_y[1].copy()
        if params.augmentations:
            xx, yy = apply_augmentation_pipeline(
                x=xx,
                y=yy,
                frame_size=params.frame_size,
                sample_rate=params.sampling_rate,
                class_map=params.class_map,
                augmentations=params.augmentations,
            )
            # xx_mu, xx_sd = np.nanmean(xx), np.nanstd(xx)
            # # Standard augmentations dont impact the label
            # xx = augment_pipeline(xx, augmentations=params.augmentations, sample_rate=params.sampling_rate)

            # # Augmentations that impact the label
            # noise_label = params.class_map.get(HeartSegment.noise, None)
            # if noise_label:

            #     cutout = next(filter(lambda a: a.name == "cutout", params.augmentations), None)
            #     if cutout:
            #         prob = cutout.params.get("probability", [0, 0.25])[1]
            #         width = cutout.params.get("width", [0, 1])
            #         if np.random.rand() < prob:
            #             dur = int(np.random.uniform(width[0], width[1])*feat_shape[0])
            #             start = np.random.randint(0, feat_shape[0] - dur)
            #             stop = start + dur
            #             xx[start:stop] = xx_mu
            #             yy[start:stop] = noise_label
            #         # END IF
            #     # END IF

            #     whiteout = next(filter(lambda a: a.name == "whiteout", params.augmentations), None)
            #     if whiteout:
            #         prob = cutout.params.get("probability", [0, 0.25])[1]
            #         amp = whiteout.params.get("amplitude", [0.5, 1.0])
            #         width = cutout.params.get("width", [0, 1])
            #         if np.random.rand() < prob:
            #             dur = int(np.random.uniform(width[0], width[1])*feat_shape[0])
            #             start = np.random.randint(0, feat_shape[0] - dur)
            #             stop = start + dur
            #             scale = np.random.uniform(amp[0], amp[1])*xx_sd
            #             xx[start:stop] += np.random.normal(0, scale, size=xx[start:stop].shape)
            #             yy[start:stop] = noise_label
            #         # END IF
            #     # END IF
            # # END IF
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
    datasets: list[HKDataset],
    params: HKTestParams | HKExportParams,
) -> tuple[npt.NDArray, npt.NDArray]:
    """Load test datasets.

    Args:
        datasets (list[HeartKitDataset]): Datasets
        params (HKTestParams|HKExportParams): Test params

    Returns:
        tuple[npt.NDArray, npt.NDArray]: Test data and labels
    """

    feat_shape = get_feat_shape(params.frame_size)

    def preprocess(x_y: tuple[npt.NDArray, npt.NDArray]) -> tuple[npt.NDArray, npt.NDArray]:
        xx = x_y[0].copy().squeeze()
        yy = x_y[1].copy()
        # if params.augmentations:
        #     xx = augment_pipeline(xx, augmentations=params.augmentations, sample_rate=params.sampling_rate)
        xx = prepare(xx, sample_rate=params.sampling_rate, preprocesses=params.preprocesses).reshape(feat_shape)
        yy = tf.one_hot(yy, params.num_classes)
        return xx, yy

    with console.status("[bold green] Loading test dataset..."):
        test_datasets = [
            ds.load_test_dataset(
                test_pt_samples=params.test_samples_per_patient,
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
    """Reference model

    Args:
        inputs (tf.Tensor): Model inputs
        num_classes (int): Number of classes

    Returns:
        keras.Model: Model
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
