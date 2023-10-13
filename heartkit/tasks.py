from typing import Any

import tensorflow as tf

from .defines import HeartTask
from .models import (
    EfficientNetParams,
    EfficientNetV2,
    MBConvParams,
    MultiresNet,
    MultiresNetParams,
    ResNet,
    ResNetParams,
    UNet,
    UNetBlockParams,
    UNetParams,
    UNext,
    UNextParams,
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
        return ["NORMAL", "PAC", "PVC"]  # , "NOISE"]
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
        return (frame_size, 1), (num_classes,)

    if task == HeartTask.beat:
        return (frame_size, 3), (num_classes,)

    if task == HeartTask.segmentation:
        return (frame_size, 1), (frame_size, num_classes)

    raise ValueError(f"unknown task: {task}")


def get_task_spec(task: HeartTask, frame_size: int) -> tuple[tf.TensorSpec, tf.TensorSpec]:
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
    inputs: tf.Tensor,
    task: HeartTask,
    name: str | None = None,
    params: dict[str, Any] | None = None,
) -> tf.keras.Model:
    """Generate model for given task

    Args:
        inputs (tf.Tensor): Model inputs
        task (HeartTask): Heart task
        name (str | None, optional): Architecture type. Defaults to None.
        params (dict[str, Any] | None, optional): Model parameters. Defaults to None.

    Returns:
        tf.keras.Model: Model
    """
    if name is None:
        return get_default_model(inputs=inputs, task=task)
    # END IF

    num_classes = get_num_classes(task=task)
    match name:
        case "resnet":
            return ResNet(x=inputs, params=ResNetParams.parse_obj(params), num_classes=num_classes)
        case "efficientnet":
            return EfficientNetV2(
                x=inputs,
                params=EfficientNetParams.parse_obj(params),
                num_classes=num_classes,
            )

        case "multiresnet":
            return MultiresNet(
                x=inputs,
                params=MultiresNetParams.parse_obj(params),
                num_classes=num_classes,
            )

        case "unet":
            return UNet(
                x=inputs,
                params=UNetParams.parse_obj(params),
                num_classes=num_classes,
            )

        case "unext":
            return UNext(
                x=inputs,
                params=UNextParams.parse_obj(params),
                num_classes=num_classes,
            )
        case _:
            raise ValueError(f"No network architecture with name {name}")
    # END MATCH


def get_default_model(
    inputs: tf.Tensor,
    task: HeartTask,
):
    """Get default model for given task"""
    num_classes = get_num_classes(task=task)
    match task:
        case HeartTask.segmentation:
            return get_segmentation_model(inputs=inputs, num_classes=num_classes)
        case HeartTask.arrhythmia:
            return get_arrhythmia_model(inputs=inputs, num_classes=num_classes)
        case HeartTask.beat:
            return get_beat_model(inputs=inputs, num_classes=num_classes)
        case _:
            raise ValueError(f"No default model for task {task}")
    # END MATCH


def get_beat_model(inputs: tf.Tensor, num_classes: int) -> tf.keras.Model:
    """Reference beat model"""
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


def get_arrhythmia_model(inputs: tf.Tensor, num_classes: int) -> tf.keras.Model:
    """Reference arrhythmia model"""
    blocks = [
        MBConvParams(
            filters=32,
            depth=2,
            ex_ratio=1,
            kernel_size=(1, 3),
            strides=(1, 2),
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
            depth=2,
            ex_ratio=1,
            kernel_size=(1, 3),
            strides=(1, 2),
            se_ratio=4,
        ),
        MBConvParams(
            filters=96,
            depth=2,
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
            input_kernel_size=(1, 5),
            input_strides=(1, 2),
            blocks=blocks,
            output_filters=0,
            include_top=True,
            dropout=0.0,
            drop_connect_rate=0.0,
        ),
        num_classes=num_classes,
    )


def get_segmentation_model(
    inputs: tf.Tensor,
    num_classes: int,
) -> tf.keras.Model:
    """Reference segmentation model"""
    blocks = [
        UNetBlockParams(filters=8, depth=2, ddepth=1, kernel=(1, 3), strides=(1, 2), skip=True),
        UNetBlockParams(filters=16, depth=2, ddepth=1, kernel=(1, 3), strides=(1, 2), skip=True),
        UNetBlockParams(filters=24, depth=2, ddepth=1, kernel=(1, 3), strides=(1, 2), skip=True),
        UNetBlockParams(filters=32, depth=2, ddepth=1, kernel=(1, 3), strides=(1, 2), skip=True),
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
    # blocks = [
    #     UNetBlockParams(
    #         filters=8, depth=2, kernel=(1, 5), strides=(1, 2), skip=True, seperable=True, norm="layer", dilation=(1, 2)
    #     ),
    #     UNetBlockParams(
    #         filters=16, depth=2, kernel=(1, 5), strides=(1, 2), skip=True, seperable=True, norm="layer", dilation=(1, 2)
    #     ),
    #     UNetBlockParams(
    #         filters=24, depth=2, kernel=(1, 5), strides=(1, 2), skip=True, seperable=True, norm="layer", dilation=(1, 2)
    #     ),
    #     UNetBlockParams(
    #         filters=32,
    #         depth=2,
    #         kernel=(1, 5),
    #         strides=(1, 2),
    #         skip=True,
    #         seperable=False,
    #         norm="layer",
    #         dilation=(1, 2),
    #     ),
    # ]
    # return UNet(
    #     inputs,
    #     params=UNetParams(
    #         blocks=blocks,
    #         output_kernel_size=(1, 5),
    #         include_top=True,
    #     ),
    #     num_classes=num_classes,
    # )
    # blocks = [
    #     UNextBlockParams(filters=8, depth=2, kernel=3, pool=2, strides=2, skip=True, expand_ratio=1, se_ratio=1, dropout=0),
    #     UNextBlockParams(filters=16, depth=2, kernel=3, pool=2, strides=2, skip=True, expand_ratio=1, se_ratio=1, dropout=0),
    #     UNextBlockParams(filters=24, depth=2, kernel=3, pool=2, strides=2, skip=True, expand_ratio=1, se_ratio=1, dropout=0),
    #     UNextBlockParams(filters=32, depth=2, kernel=53, pool=2, strides=2, skip=True, expand_ratio=1, se_ratio=1, dropout=0),
    # ]
    # return UNext(
    #     inputs,
    #     params=UNextParams(
    #         blocks=blocks,
    #         output_kernel_size=3,
    #         include_top=True,
    #     ),
    #     num_classes=num_classes,
    # )
