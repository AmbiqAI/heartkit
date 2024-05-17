import keras
import tensorflow as tf
from rich.console import Console

from ...defines import ModelArchitecture
from ...models import ModelFactory, UNet, UNetBlockParams, UNetParams

console = Console()


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

    return default_model(inputs=inputs, num_classes=num_classes)


def default_model(
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
