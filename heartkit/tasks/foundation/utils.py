from abc import abstractmethod
from pathlib import Path
from typing import Callable

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
from ...models import ModelFactory

console = Console()


class ContrastiveModel(keras.Model):
    """Base class for contrastive learning models"""

    def __init__(
        self,
        encoder: keras.Model,
        projector: keras.Model,
        contrastive_augmenter: Callable[[tf.Tensor], tf.Tensor] | None = None,
        classification_augmenter: Callable[[tf.Tensor], tf.Tensor] | None = None,
        linear_probe: keras.Model | None = None,
    ):
        super().__init__()

        self.encoder = encoder
        self.projector = projector
        self.contrastive_augmenter = contrastive_augmenter
        self.classification_augmenter = classification_augmenter
        self.linear_probe = linear_probe

        self.probe_loss = None
        self.probe_optimizer = None
        self.contrastive_loss_tracker = None
        self.contrastive_optimizer = None
        self.contrastive_accuracy = None
        self.correlation_accuracy = None
        self.probe_accuracy = None

    @property
    def metrics(self):
        """List of metrics to track during training and evaluation"""
        return [
            self.contrastive_loss_tracker,
            self.correlation_accuracy,
            self.contrastive_accuracy,
            # self.probe_loss_tracker,
            # self.probe_accuracy,
        ]

    @abstractmethod
    def contrastive_loss(self, projections_1, projections_2):
        """Contrastive loss function"""
        raise NotImplementedError()

    def call(self, inputs, training=None, mask=None):
        """Forward pass through the encoder model"""
        return self.encoder(inputs, training=training, mask=mask)

    # pylint: disable=unused-argument,arguments-differ
    def compile(
        self,
        contrastive_optimizer: keras.optimizers.Optimizer,
        probe_optimizer: keras.optimizers.Optimizer | None = None,
        **kwargs
    ):
        """Compile the model with the specified optimizers"""
        super().compile(**kwargs)

        self.contrastive_optimizer = contrastive_optimizer
        self.probe_optimizer = probe_optimizer

        # self.contrastive_loss is a method that will be implemented by the subclasses
        self.probe_loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

        self.contrastive_loss_tracker = keras.metrics.Mean(name="loss")
        self.contrastive_accuracy = keras.metrics.SparseCategoricalAccuracy(name="c_acc")
        self.correlation_accuracy = keras.metrics.SparseCategoricalAccuracy(name="r_acc")

        self.probe_accuracy = keras.metrics.SparseCategoricalAccuracy()

    def save(self, filepath, overwrite=True, save_format=None, **kwargs):
        """Save the encoder model to file

        Args:
            filepath (str): Filepath
            overwrite (bool, optional): Overwrite existing file. Defaults to True.
            save_format ([type], optional): Save format. Defaults to None.
        """
        self.encoder.save(filepath, overwrite, save_format, **kwargs)

    def reset_metrics(self):
        """Reset the metrics to their initial state"""
        self.contrastive_accuracy.reset_states()
        self.correlation_accuracy.reset_states()
        self.probe_accuracy.reset_states()

    def update_contrastive_accuracy(self, features_1, features_2):
        """Update the contrastive accuracy metric
        self-supervised metric inspired by the SimCLR loss
        """

        # cosine similarity: the dot product of the l2-normalized feature vectors
        features_1 = tf.math.l2_normalize(features_1, axis=1)
        features_2 = tf.math.l2_normalize(features_2, axis=1)
        similarities = tf.matmul(features_1, features_2, transpose_b=True)

        # Push positive pairs to the diagonal
        batch_size = tf.shape(features_1)[0]
        contrastive_labels = tf.range(batch_size)
        self.contrastive_accuracy.update_state(contrastive_labels, similarities)
        self.contrastive_accuracy.update_state(contrastive_labels, tf.transpose(similarities))

    def update_correlation_accuracy(self, features_1, features_2):
        """Update the correlation accuracy metric
        self-supervised metric inspired by the BarlowTwins loss
        """

        # normalization so that cross-correlation will be between -1 and 1
        features_1 = (features_1 - tf.reduce_mean(features_1, axis=0)) / tf.math.reduce_std(features_1, axis=0)
        features_2 = (features_2 - tf.reduce_mean(features_2, axis=0)) / tf.math.reduce_std(features_2, axis=0)

        # the cross correlation of image representations should be the identity matrix
        batch_size = tf.shape(features_1, out_type=tf.int32)[0]
        batch_size = tf.cast(batch_size, tf.float32)
        cross_correlation = tf.matmul(features_1, features_2, transpose_a=True) / batch_size

        feature_dim = tf.shape(features_1)[1]
        correlation_labels = tf.range(feature_dim)
        self.correlation_accuracy.update_state(correlation_labels, cross_correlation)
        self.correlation_accuracy.update_state(correlation_labels, tf.transpose(cross_correlation))

    def train_step(self, data):
        """Training step for the model"""
        pair1, pair2 = data

        # each input is augmented twice, differently
        augmented_inputs_1 = self.contrastive_augmenter(pair1)
        augmented_inputs_2 = self.contrastive_augmenter(pair2)
        with tf.GradientTape() as tape:
            # Encoder phase
            features_1 = self.encoder(augmented_inputs_1)
            features_2 = self.encoder(augmented_inputs_2)
            # Projection phase
            projections_1 = self.projector(features_1)
            projections_2 = self.projector(features_2)
            contrastive_loss = self.contrastive_loss(projections_1, projections_2)
        # END WITH

        # backpropagation
        gradients = tape.gradient(
            contrastive_loss,
            self.encoder.trainable_weights + self.projector.trainable_weights,
        )
        self.contrastive_optimizer.apply_gradients(
            zip(
                gradients,
                self.encoder.trainable_weights + self.projector.trainable_weights,
            )
        )

        self.contrastive_loss_tracker.update_state(contrastive_loss)

        self.update_contrastive_accuracy(features_1, features_2)
        self.update_correlation_accuracy(features_1, features_2)

        # # labels are only used in evalutation for probing
        # augmented_inputs = self.classification_augmenter(labeled_pair)
        # with tf.GradientTape() as tape:
        #     features = self.encoder(augmented_inputs)
        #     class_logits = self.linear_probe(features)
        #     probe_loss = self.probe_loss(labels, class_logits)
        # gradients = tape.gradient(probe_loss, self.linear_probe.trainable_weights)
        # self.probe_optimizer.apply_gradients(
        #     zip(gradients, self.linear_probe.trainable_weights)
        # )
        # self.probe_accuracy.update_state(labels, class_logits)

        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        """Test step for the model"""
        pair1, pair2 = data
        augmented_inputs_1 = self.contrastive_augmenter(pair1)
        augmented_inputs_2 = self.contrastive_augmenter(pair2)
        features_1 = self.encoder(augmented_inputs_1, training=False)
        features_2 = self.encoder(augmented_inputs_2, training=False)
        projections_1 = self.projector(features_1, training=False)
        projections_2 = self.projector(features_2, training=False)

        contrastive_loss = self.contrastive_loss(projections_1, projections_2)
        self.contrastive_loss_tracker.update_state(contrastive_loss)
        self.update_contrastive_accuracy(features_1, features_2)
        self.update_correlation_accuracy(features_1, features_2)

        return {m.name: m.result() for m in self.metrics}


class SimCLR(ContrastiveModel):
    """SimCLR model for self-supervised learning"""

    def __init__(
        self,
        encoder: keras.Model,
        projector: keras.Model,
        contrastive_augmenter: Callable[[tf.Tensor], tf.Tensor] | None = None,
        classification_augmenter: Callable[[tf.Tensor], tf.Tensor] | None = None,
        linear_probe: keras.Model | None = None,
        temperature: float = 0.1,
    ):
        super().__init__(
            encoder=encoder,
            projector=projector,
            contrastive_augmenter=contrastive_augmenter,
            classification_augmenter=classification_augmenter,
            linear_probe=linear_probe,
        )
        self.temperature = temperature

    def contrastive_loss(self, projections_1, projections_2):
        """Contrastive loss function for SimCLR"""
        # InfoNCE loss (information noise-contrastive estimation)
        # NT-Xent loss (normalized temperature-scaled cross entropy)

        # cosine similarity: the dot product of the l2-normalized feature vectors
        projections_1 = tf.math.l2_normalize(projections_1, axis=1)
        projections_2 = tf.math.l2_normalize(projections_2, axis=1)
        similarities = tf.matmul(projections_1, projections_2, transpose_b=True) / self.temperature

        # the temperature-scaled similarities are used as logits for cross-entropy
        batch_size = tf.shape(projections_1)[0]
        contrastive_labels = tf.range(batch_size)
        loss1 = keras.losses.sparse_categorical_crossentropy(contrastive_labels, similarities, from_logits=True)
        loss2 = keras.losses.sparse_categorical_crossentropy(
            contrastive_labels, tf.transpose(similarities), from_logits=True
        )
        return (loss1 + loss2) / 2


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
    return (frame_size,)


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
    spec: tuple[tf.TensorSpec, tf.TensorSpec],
    class_map: dict[int, int],
    datasets: list[DatasetParams],
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
                    task="foundation",
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
        tuple[tf.data.Dataset, tf.data.Dataset]: Train and validation datasets
    """

    feat_shape = get_feat_shape(params.frame_size)

    def preprocess(x_y: tuple[npt.NDArray, npt.NDArray]) -> tuple[npt.NDArray, npt.NDArray]:
        p1 = x_y[0].copy()
        p2 = x_y[1].copy()
        if params.augmentations:
            p1 = augment_pipeline(
                x=p1,
                augmentations=params.augmentations,
                sample_rate=params.sampling_rate,
            )
            p2 = augment_pipeline(
                x=p2,
                augmentations=params.augmentations,
                sample_rate=params.sampling_rate,
            )
        p1 = prepare(p1, sample_rate=params.sampling_rate, preprocesses=params.preprocesses).reshape(feat_shape)
        p2 = prepare(p2, sample_rate=params.sampling_rate, preprocesses=params.preprocesses).reshape(feat_shape)
        return p1, p2

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
        p1 = x_y[0].copy()
        p2 = x_y[1].copy()
        if params.augmentations:
            p1 = augment_pipeline(
                x=p1,
                augmentations=params.augmentations,
                sample_rate=params.sampling_rate,
            )
            p2 = augment_pipeline(
                x=p2,
                augmentations=params.augmentations,
                sample_rate=params.sampling_rate,
            )
        # END IF
        p1 = prepare(p1, sample_rate=params.sampling_rate, preprocesses=params.preprocesses).reshape(feat_shape)
        p2 = prepare(p2, sample_rate=params.sampling_rate, preprocesses=params.preprocesses).reshape(feat_shape)
        return p1, p2

    with console.status("[bold green] Loading test dataset..."):
        test_datasets = [
            ds.load_test_dataset(
                test_patients=params.test_patients,
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


def create_model(inputs: tf.Tensor, num_classes: int | None, architecture: ModelArchitecture | None) -> keras.Model:
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
    # END IF
    raise ValueError("No default model found")
