import tensorflow as tf
from keras.engine.keras_tensor import KerasTensor

from ..defines import ArchitectureType, HeartTask, get_num_classes
from ..models.features import ecg_feature_extractor


def create_task_model(
    inputs: KerasTensor,
    task: HeartTask,
    arch: ArchitectureType = "resnet18",
    stages: int | None = None,
) -> tf.keras.Model:
    """Generate model for given task

    Args:
        inputs (KerasTensor): Model inputs
        task (HeartTask): Heart task
        arch (ArchitectureType, optional): Architecture type. Defaults to 'resnet18'.
        stages (int | None, optional): # stages in network. Defaults to None.

    Returns:
        tf.keras.Model: Model
    """
    num_classes = get_num_classes(task=task)
    x = ecg_feature_extractor(inputs, arch, stages=stages)
    outputs = tf.keras.layers.Dense(num_classes)(x)
    model = tf.keras.Model(inputs, outputs, name="model")
    return model


def get_task_spec(task: HeartTask, frame_size: int) -> tuple[tf.TensorSpec]:
    """Get task model spec

    Args:
        task (HeartTask): ECG task
        frame_size (int): Frame size

    Returns:
        tuple[tf.TensorSpec]: TF spec for task
    """
    num_classes = get_num_classes(task)
    if task in [HeartTask.rhythm, HeartTask.hr]:
        return (
            tf.TensorSpec((frame_size, 1), tf.float32),
            tf.TensorSpec((), tf.int32),
        )
    if task in [HeartTask.beat]:
        return (
            tf.TensorSpec((frame_size, 1), tf.float32),
            tf.TensorSpec((num_classes), tf.int32),
        )
    if task in [HeartTask.segmentation]:
        return (
            tf.TensorSpec((frame_size, 1), tf.float32),
            tf.TensorSpec((frame_size, 1), tf.int32),
        )
    raise ValueError(f"unknown task: {task}")
