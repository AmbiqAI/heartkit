from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt
import tensorflow as tf
from keras.engine.keras_tensor import KerasTensor

from ..types import ArchitectureType, EcgTask
from .features import ecg_feature_extractor

InputShape = Union[Tuple[int], List[Tuple[int]], Dict[str, Tuple[int]]]


def build_input_tensor_from_shape(
    shape: InputShape, dtype: tf.DType = None, ignore_batch_dim: bool = False
):
    """Build input tensor from shape which can be used to initialize the weights of a model.

    Args:
        shape (InputShape]): Input Shape
        dtype (tf.DType, optional): _description_. Defaults to None.
        ignore_batch_dim (bool, optional): Ignore first dimension as batch. Defaults to False.

    Returns:
        tf.keras.layers.Input: Input layer
    """
    if isinstance(shape, (list, tuple)):
        return [
            build_input_tensor_from_shape(
                shape=shape[i],
                dtype=dtype[i] if dtype else None,
                ignore_batch_dim=ignore_batch_dim,
            )
            for i in range(len(shape))
        ]

    if isinstance(shape, dict):
        return {
            k: build_input_tensor_from_shape(
                shape=shape[k],
                dtype=dtype[k] if dtype else None,
                ignore_batch_dim=ignore_batch_dim,
            )
            for k in shape
        }

    if ignore_batch_dim:
        shape = shape[1:]
    return tf.keras.layers.Input(shape, dtype=dtype)


def generate_task_model(
    inputs: KerasTensor,
    task: EcgTask,
    arch: ArchitectureType = "resnet18",
    stages: Optional[int] = None,
) -> tf.keras.Model:
    """Generate model for given task

    Args:
        inputs (KerasTensor): Model inputs
        task (EcgTask): Heart task
        arch (ArchitectureType, optional): Architecture type. Defaults to 'resnet18'.
        stages (Optional[int], optional): # stages in network. Defaults to None.

    Returns:
        tf.keras.Model: Model
    """

    if task == EcgTask.rhythm:
        num_classes = 2  # 4
    elif task == EcgTask.beat:
        num_classes = 3  # 5
    elif task == EcgTask.hr:
        num_classes = 4
    elif task == EcgTask.segmentation:
        num_classes = 3
    else:
        raise ValueError(f"unknown task: {task}")
    x = ecg_feature_extractor(inputs, arch, stages=stages)
    outputs = tf.keras.layers.Dense(num_classes)(x)
    model = tf.keras.Model(inputs, outputs, name="model")
    return model


def get_pretrained_weights(
    inputs: KerasTensor,
    checkpoint_file: str,
    task: EcgTask,
    arch: ArchitectureType = "resnet18",
    stages: Optional[int] = None,
) -> tf.keras.Model:
    """Initialize model with weights from file

    Args:
        checkpoint_file (str): TensorFlow checkpoint file containing weights
        task (EcgTask): Hear task
        arch (ArchitectureType, optional): Architecture type. Defaults to 'resnet18'.
        stages (Optional[int], optional): # stages in network. Defaults to None.

    Returns:
        tf.keras.Model: Pre-trained model
    """
    model = generate_task_model(task, arch, stages=stages)
    if task in [EcgTask.rhythm, EcgTask.beat, EcgTask.hr]:
        inputs = build_input_tensor_from_shape(tf.TensorShape((None, 1)))
    else:
        raise ValueError(f"Unknown task: {task}")
    model(inputs)
    model.load_weights(checkpoint_file)
    return model


def get_predicted_threshold_indices(
    y_prob: npt.ArrayLike,
    y_pred: Optional[npt.ArrayLike] = None,
    threshold: float = 0.5,
) -> npt.ArrayLike:
    """Get prediction indices that are above threshold (confidence level).
    This is useful to remove weak prediction that can happen due to noisy data or poor model performance.

    Args:
        y_prob (npt.ArrayLike): Model output as probabilities
        y_pred (npt.ArrayLike, optional): Model predictions. Defaults to None.
        threshold (float): Confidence level

    Returns:
        npt.ArrayLike: Indices of y_prob that satisfy threshold
    """
    if y_pred is None:
        y_pred = np.argmax(y_prob, axis=1)

    y_pred_prob = np.take_along_axis(
        y_prob, np.expand_dims(y_pred, axis=-1), axis=-1
    ).squeeze(axis=-1)
    y_thresh_idx = np.where(y_pred_prob > threshold)[0]
    return y_thresh_idx


def get_strategy(use_mixed_precision: bool = False) -> tf.distribute.Strategy:
    """Select best distribution strategy.
    Args:
        use_mixed_precision (bool, optional): Use mixed precision on TPU. Defaults to False.
    Returns:
        tf.distribute.Strategy: Strategy
    """
    # Try to detect an available TPU. If none is present, default to MirroredStrategy
    try:
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver.connect()
        strategy = tf.distribute.TPUStrategy(tpu)
        if use_mixed_precision:
            tf.keras.mixed_precision.set_global_policy("mixed_bfloat16")
    except ValueError:
        # MirroredStrategy is best for a single machine with one or multiple GPUs
        strategy = tf.distribute.MirroredStrategy()
    return strategy
