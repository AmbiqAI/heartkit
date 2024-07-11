import keras
from keras_edge.models.tcn import Tcn, TcnBlockParams, TcnParams
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
        return ModelFactory.create(
            name=architecture.name,
            params=architecture.params,
            inputs=inputs,
            num_classes=num_classes,
        )

    return _default_model(inputs=inputs, num_classes=num_classes)


def _default_model(
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
    # Default model

    blocks = [
        TcnBlockParams(
            filters=8,
            kernel=(1, 7),
            dilation=(1, 1),
            dropout=0.1,
            ex_ratio=1,
            se_ratio=0,
            norm="batch",
        ),
        TcnBlockParams(
            filters=12,
            kernel=(1, 7),
            dilation=(1, 1),
            dropout=0.1,
            ex_ratio=1,
            se_ratio=2,
            norm="batch",
        ),
        TcnBlockParams(
            filters=16,
            kernel=(1, 7),
            dilation=(1, 2),
            dropout=0.1,
            ex_ratio=1,
            se_ratio=2,
            norm="batch",
        ),
        TcnBlockParams(
            filters=24,
            kernel=(1, 7),
            dilation=(1, 4),
            dropout=0.1,
            ex_ratio=1,
            se_ratio=2,
            norm="batch",
        ),
        TcnBlockParams(
            filters=32,
            kernel=(1, 7),
            dilation=(1, 8),
            dropout=0.1,
            ex_ratio=1,
            se_ratio=2,
            norm="batch",
        ),
    ]

    return Tcn(
        x=inputs,
        params=TcnParams(
            input_kernel=(1, 7),
            input_norm="batch",
            blocks=blocks,
            output_kernel=(1, 7),
            include_top=True,
            use_logits=True,
            model_name="tcn",
        ),
        num_classes=num_classes,
    )
