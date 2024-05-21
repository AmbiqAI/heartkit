from abc import abstractmethod
from typing import Callable

import keras
import tensorflow as tf


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

    def update_contrastive_accuracy(self, features_1, features_2):
        """Update the contrastive accuracy metric

        Args:
            features_1 (tf.Tensor): Features from the first augmented view
            features_2 (tf.Tensor): Features from the second augmented view
        """
        # self-supervised metric inspired by the SimCLR loss

        # cosine similarity: the dot product of the l2-normalized feature vectors
        features_1 = tf.math.l2_normalize(features_1, axis=1)
        features_2 = tf.math.l2_normalize(features_2, axis=1)
        similarities = tf.matmul(features_1, features_2, transpose_b=True)

        # the similarity between the representations of two augmented views of the
        # same image should be higher than their similarity with other views
        batch_size = tf.shape(features_1)[0]
        contrastive_labels = tf.range(batch_size)
        self.contrastive_accuracy.update_state(
            tf.concat([contrastive_labels, contrastive_labels], axis=0),
            tf.concat([similarities, tf.transpose(similarities)], axis=0),
        )

    def update_correlation_accuracy(self, features_1, features_2):
        """Update the correlation accuracy metric
        # self-supervised metric inspired by the BarlowTwins loss

        Args:
            features_1 (tf.Tensor): Features from the first augmented view
            features_2 (tf.Tensor): Features from the second augmented view
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
        self.correlation_accuracy.update_state(
            tf.concat([correlation_labels, correlation_labels], axis=0),
            tf.concat([cross_correlation, tf.transpose(cross_correlation)], axis=0),
        )

    @abstractmethod
    def contrastive_loss(self, projections_1, projections_2):
        """Contrastive loss function"""
        raise NotImplementedError()

    def correlation_loss(self, features_1, features_2):
        """Correlation loss function"""

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
            self.correlation_loss(features_1, features_2)
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

        self.update_contrastive_accuracy(projections_1, projections_2)
        self.update_correlation_accuracy(projections_1, projections_2)
        self.contrastive_loss_tracker.update_state(contrastive_loss)

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
        self.correlation_loss(features_1, features_2)

        self.update_contrastive_accuracy(projections_1, projections_2)
        self.update_correlation_accuracy(projections_1, projections_2)
        self.contrastive_loss_tracker.update_state(contrastive_loss)

        return {m.name: m.result() for m in self.metrics}


class MomentumContrastiveModel(ContrastiveModel):
    """Base class for momentum contrastive learning models"""

    def __init__(
        self,
        encoder: keras.Model,
        projector: keras.Model,
        contrastive_augmenter: Callable[[tf.Tensor], tf.Tensor] | None = None,
        classification_augmenter: Callable[[tf.Tensor], tf.Tensor] | None = None,
        linear_probe: keras.Model | None = None,
        momentum_coeff: float = 0.999,
    ):
        super().__init__(
            encoder=encoder,
            projector=projector,
            contrastive_augmenter=contrastive_augmenter,
            classification_augmenter=classification_augmenter,
            linear_probe=linear_probe,
        )
        self.momentum_coeff = momentum_coeff

        # the momentum networks are initialized from their online counterparts
        self.m_encoder = keras.models.clone_model(self.encoder)
        self.m_projector = keras.models.clone_model(self.projector)

    @abstractmethod
    def contrastive_loss(
        self,
        projections_1,
        projections_2,
        m_projections_1,
        m_projections_2,
    ):  # pylint: disable=arguments-differ
        pass

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
            # Momentum encoder phase
            m_features_1 = self.m_encoder(augmented_inputs_1)
            m_features_2 = self.m_encoder(augmented_inputs_2)
            # Momentum projection phase
            m_projections_1 = self.m_projector(m_features_1)
            m_projections_2 = self.m_projector(m_features_2)
            contrastive_loss = self.contrastive_loss(projections_1, projections_2, m_projections_1, m_projections_2)
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
        self.correlation_loss(m_features_1, m_features_2)

        self.update_contrastive_accuracy(m_features_1, m_features_2)
        self.update_correlation_accuracy(m_features_1, m_features_2)

        # labeled_inputs = None
        # labels = None
        # preprocessed_inputs = self.classification_augmenter(labeled_inputs)
        # with tf.GradientTape() as tape:
        #     # the momentum encoder is used here as it moves more slowly
        #     features = self.m_encoder(preprocessed_inputs)
        #     class_logits = self.linear_probe(features)
        #     probe_loss = self.probe_loss(labels, class_logits)
        # gradients = tape.gradient(probe_loss, self.linear_probe.trainable_weights)
        # self.probe_optimizer.apply_gradients(
        #     zip(gradients, self.linear_probe.trainable_weights)
        # )
        # self.probe_accuracy.update_state(labels, class_logits)

        # the momentum networks are updated by exponential moving average
        for weight, m_weight in zip(self.encoder.weights, self.m_encoder.weights):
            m_weight.assign(self.momentum_coeff * m_weight + (1 - self.momentum_coeff) * weight)
        for weight, m_weight in zip(self.projector.weights, self.m_projector.weights):
            m_weight.assign(self.momentum_coeff * m_weight + (1 - self.momentum_coeff) * weight)

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
        m_features_1 = self.m_encoder(augmented_inputs_1, training=False)
        m_features_2 = self.m_encoder(augmented_inputs_2, training=False)
        m_projections_1 = self.m_projector(m_features_1, training=False)
        m_projections_2 = self.m_projector(m_features_2, training=False)

        contrastive_loss = self.contrastive_loss(projections_1, projections_2, m_projections_1, m_projections_2)

        self.contrastive_loss_tracker.update_state(contrastive_loss)
        self.correlation_loss(m_features_1, m_features_2)

        self.update_contrastive_accuracy(m_features_1, m_features_2)
        self.update_correlation_accuracy(m_features_1, m_features_2)

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

        # Cross-entropy loss
        loss1 = keras.losses.sparse_categorical_crossentropy(contrastive_labels, similarities, from_logits=True)
        loss2 = keras.losses.sparse_categorical_crossentropy(
            contrastive_labels, tf.transpose(similarities), from_logits=True
        )
        return (loss1 + loss2) / 2


class MoCo(MomentumContrastiveModel):
    """MoCo model for self-supervised learning"""

    def __init__(
        self,
        encoder: keras.Model,
        projector: keras.Model,
        contrastive_augmenter: Callable[[tf.Tensor], tf.Tensor] | None = None,
        classification_augmenter: Callable[[tf.Tensor], tf.Tensor] | None = None,
        linear_probe: keras.Model | None = None,
        momentum_coeff: float = 0.999,
        temperature: float = 0.1,
        queue_size: int = 65536,
    ):
        super().__init__(
            encoder=encoder,
            projector=projector,
            contrastive_augmenter=contrastive_augmenter,
            classification_augmenter=classification_augmenter,
            linear_probe=linear_probe,
            momentum_coeff=momentum_coeff,
        )
        self.temperature = temperature

        feature_dimensions = encoder.output_shape[1]
        self.feature_queue = tf.Variable(
            tf.math.l2_normalize(tf.random.normal(shape=(queue_size, feature_dimensions)), axis=1),
            trainable=False,
        )

    def contrastive_loss(
        self,
        projections_1,
        projections_2,
        m_projections_1,
        m_projections_2,
    ):
        # similar to the SimCLR loss, however it uses the momentum networks'
        # representations of the differently augmented views as targets
        projections_1 = tf.math.l2_normalize(projections_1, axis=1)
        projections_2 = tf.math.l2_normalize(projections_2, axis=1)
        m_projections_1 = tf.math.l2_normalize(m_projections_1, axis=1)
        m_projections_2 = tf.math.l2_normalize(m_projections_2, axis=1)

        similarities_1_2 = (
            tf.matmul(
                projections_1,
                tf.concat((m_projections_2, self.feature_queue), axis=0),
                transpose_b=True,
            )
            / self.temperature
        )
        similarities_2_1 = (
            tf.matmul(
                projections_2,
                tf.concat((m_projections_1, self.feature_queue), axis=0),
                transpose_b=True,
            )
            / self.temperature
        )

        batch_size = tf.shape(projections_1)[0]
        contrastive_labels = tf.range(batch_size)
        loss = keras.losses.sparse_categorical_crossentropy(
            tf.concat([contrastive_labels, contrastive_labels], axis=0),
            tf.concat([similarities_1_2, similarities_2_1], axis=0),
            from_logits=True,
        )

        # feature queue update
        self.feature_queue.assign(
            tf.concat(
                [
                    m_projections_1,
                    m_projections_2,
                    self.feature_queue[: -(2 * batch_size)],
                ],
                axis=0,
            )
        )
        return loss
