import keras
from keras_edge.models.efficientnet import (
    EfficientNetParams,
    EfficientNetV2,
    MBConvParams,
)

from ...defines import ModelArchitecture
from ...models import ModelFactory


def create_model(inputs: keras.KerasTensor, num_classes: int, architecture: ModelArchitecture | None) -> keras.Model:
    """Generate model or use default

    Args:
        inputs (keras.KerasTensor): Model inputs
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

    return default_model(inputs=inputs, num_classes=num_classes)


def default_model(
    inputs: keras.KerasTensor,
    num_classes: int,
) -> keras.Model:
    """Reference model

    Args:
        inputs (keras.KerasTensor): Model inputs
        num_classes (int): Number of classes

    Returns:
        keras.Model: Model
    """

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
            depth=1,
            ex_ratio=1,
            kernel_size=(1, 3),
            strides=(1, 2),
            se_ratio=4,
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
            filters=80,
            depth=1,
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
            input_kernel_size=(1, 3),
            input_strides=(1, 2),
            blocks=blocks,
            output_filters=0,
            include_top=True,
            dropout=0.0,
            drop_connect_rate=0.0,
        ),
        num_classes=num_classes,
    )
