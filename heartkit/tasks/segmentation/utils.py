import keras
from neuralspot_edge.models.unet import UNet, UNetBlockParams, UNetParams
from rich.console import Console

from ...defines import ModelArchitecture
from ...models import ModelFactory

console = Console()


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
        return ModelFactory.get(architecture.name)(
            x=inputs,
            params=architecture.params,
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
