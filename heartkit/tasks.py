from typing import Any

import tensorflow as tf
from keras.engine.keras_tensor import KerasTensor

from .defines import HeartTask
from .models import (
    EfficientNetParams,
    EfficientNetV2,
    MBConvParams,
    ResNet,
    ResNetParams,
    UNet,
    UNetBlockParams,
    UNetParams,
)


def get_class_names(task: HeartTask) -> list[str]:
    """Get class names for given task

    Args:
        task (HeartTask): Heart task

    Returns:
        list[str]: class names
    """
    if task == HeartTask.arrhythmia:
        # NOTE: Bucket AFIB and AFL together
        return ["NSR", "AFIB/AFL"]
    if task == HeartTask.beat:
        return ["NORMAL", "PAC", "PVC"]
    if task == HeartTask.hrv:
        return ["NORMAL", "TACHYCARDIA", "BRADYCARDIA"]
    if task == HeartTask.segmentation:
        return ["NONE", "P-WAVE", "QRS", "T-WAVE"]
    raise ValueError(f"unknown task: {task}")


def get_num_classes(task: HeartTask) -> int:
    """Get number of classes for given task

    Args:
        task (HeartTask): Heart task

    Returns:
        int: # classes
    """
    return len(get_class_names(task=task))


def get_task_shape(task: HeartTask, frame_size: int) -> tuple[tuple[int], tuple[int]]:
    """Get task model spec

    Args:
        task (HeartTask): Heart task
        frame_size (int): Frame size

    Returns:
        tuple[tuple[int], tuple[int]]: Input shape
    """
    num_classes = get_num_classes(task)
    if task == HeartTask.arrhythmia:
        return (1, frame_size, 1), (num_classes,)

    if task == HeartTask.beat:
        return (1, frame_size, 3), (num_classes,)

    if task == HeartTask.segmentation:
        return (1, frame_size, 1), (frame_size, num_classes)

    raise ValueError(f"unknown task: {task}")


def get_task_spec(
    task: HeartTask, frame_size: int
) -> tuple[tf.TensorSpec, tf.TensorSpec]:
    """Get task model spec

    Args:
        task (HeartTask): Heart task
        frame_size (int): Frame size

    Returns:
        tuple[tf.TensorSpec]: TF spec for task
    """
    in_shape, out_shape = get_task_shape(task, frame_size)
    return (
        tf.TensorSpec(in_shape, tf.float32),
        tf.TensorSpec(out_shape, tf.int32),
    )


def create_task_model(
    inputs: KerasTensor,
    task: HeartTask,
    name: str | None = None,
    params: dict[str, Any] | None = None,
) -> tf.keras.Model:
    """Generate model for given task

    Args:
        inputs (KerasTensor): Model inputs
        task (HeartTask): Heart task
        name (str | None, optional): Architecture type. Defaults to None.
        params (dict[str, Any] | None, optional): Model parameters. Defaults to None.

    Returns:
        tf.keras.Model: Model
    """
    num_classes = get_num_classes(task=task)
    if name == "resnet":
        return ResNet(
            x=inputs, params=ResNetParams.parse_obj(params), num_classes=num_classes
        )
    if name == "efficientnet":
        return EfficientNetV2(
            x=inputs,
            params=EfficientNetParams.parse_obj(params),
            num_classes=num_classes,
        )
    if name:
        raise ValueError(f"No network architecture with name {name}")

    # Otherwise use reference model
    if task == HeartTask.segmentation:
        return get_segmentation_model(inputs=inputs, num_classes=num_classes)

    if task == HeartTask.arrhythmia:
        return get_arrhythmia_model(inputs=inputs, num_classes=num_classes)

    if task == HeartTask.beat:
        return get_beat_model(inputs=inputs, num_classes=num_classes)

    raise NotImplementedError()


def get_beat_model(inputs: KerasTensor, num_classes: int) -> tf.keras.Model:
    """Reference beat model"""
    blocks = [
        MBConvParams(
            filters=32,
            depth=3,
            ex_ratio=1,
            kernel_size=(1, 5),
            strides=(1, 2),
            se_ratio=2,
        ),
        MBConvParams(
            filters=48,
            depth=3,
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
            input_kernel_size=(1, 7),
            output_filters=0,
            blocks=blocks,
            include_top=True,
            dropout=0.2,
            drop_connect_rate=0.0,
        ),
        num_classes=num_classes,
    )


def get_arrhythmia_model(inputs: KerasTensor, num_classes: int) -> tf.keras.Model:
    """Reference arrhythmia model"""
    blocks = [
        MBConvParams(
            filters=32,
            depth=3,
            ex_ratio=1,
            kernel_size=(1, 5),
            strides=(1, 2),
            se_ratio=2,
        ),
        MBConvParams(
            filters=48,
            depth=3,
            ex_ratio=1,
            kernel_size=(1, 3),
            strides=(1, 2),
            se_ratio=4,
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
            input_kernel_size=(1, 7),
            input_strides=(1, 2),
            blocks=blocks,
            output_filters=0,
            include_top=True,
            dropout=0.2,
            drop_connect_rate=0.2,
        ),
        num_classes=num_classes,
    )


def get_segmentation_model(
    inputs: KerasTensor,
    num_classes: int,
) -> tf.keras.Model:
    """Reference segmentation model"""
    blocks = [
        UNetBlockParams(filters=16, depth=1, kernel=(1, 3), strides=(1, 2), skip=False),
        UNetBlockParams(filters=32, depth=1, kernel=(1, 3), strides=(1, 2), skip=True),
        UNetBlockParams(filters=48, depth=1, kernel=(1, 3), strides=(1, 2), skip=True),
        UNetBlockParams(filters=64, depth=1, kernel=(1, 3), strides=(1, 2), skip=True),
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
