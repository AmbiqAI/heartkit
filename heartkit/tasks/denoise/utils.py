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
from ...models import ModelFactory  # , Tcn, TcnBlockParams, TcnParams

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
                    task="denoise",
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
    """Load train datasets.

    Args:
        datasets (list[HeartKitDataset]): Datasets
        params (HKTrainParams): Train params

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

    def preprocess(x: npt.NDArray) -> npt.NDArray:
        xx = x.copy().squeeze()
        if params.augmentations:
            xx = augment_pipeline(xx, augmentations=params.augmentations, sample_rate=params.sampling_rate)
        xx = prepare(xx, sample_rate=params.sampling_rate, preprocesses=params.preprocesses)
        return xx

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
    # Default model

    y = inputs
    y = keras.layers.Reshape((1,) + y.shape[1:])(y)

    y = keras.layers.Conv2D(filters=24, kernel_size=(1, 5), padding="same")(y)
    y = keras.layers.BatchNormalization()(y)
    y = keras.layers.Activation("tanh")(y)

    y = keras.layers.Reshape(y.shape[2:])(y)

    y = keras.layers.Bidirectional(keras.layers.LSTM(units=48, return_sequences=True))(y)
    y = keras.layers.BatchNormalization()(y)
    y = keras.layers.Activation("tanh")(y)

    # y = keras.layers.TimeDistributed(keras.layers.Dense(64))(y)
    # y = keras.layers.LayerNormalization(axis=[1])(y)
    # y = keras.layers.Activation("tanh")(y)

    y = keras.layers.TimeDistributed(keras.layers.Dense(num_classes))(y)
    y = keras.layers.Activation("tanh")(y)

    model = keras.models.Model(inputs, y)
    return model

    # blocks = [
    #     TcnBlockParams(filters=8, kernel=(1, 3), dilation=(1, 1), dropout=0.1, ex_ratio=1, se_ratio=0, norm="batch"),
    #     TcnBlockParams(filters=16, kernel=(1, 3), dilation=(1, 2), dropout=0.1, ex_ratio=1, se_ratio=0, norm="batch"),
    #     TcnBlockParams(filters=24, kernel=(1, 3), dilation=(1, 4), dropout=0.1, ex_ratio=1, se_ratio=4, norm="batch"),
    #     TcnBlockParams(filters=32, kernel=(1, 3), dilation=(1, 8), dropout=0.1, ex_ratio=1, se_ratio=4, norm="batch"),
    # ]

    # return Tcn(
    #     x=inputs,
    #     params=TcnParams(
    #         input_kernel=(1, 3),
    #         input_norm="batch",
    #         blocks=blocks,
    #         output_kernel=(1, 3),
    #         include_top=True,
    #         use_logits=True,
    #         model_name="tcn",
    #     ),
    #     num_classes=num_classes,
    # )
