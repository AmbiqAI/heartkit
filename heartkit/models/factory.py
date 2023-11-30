from typing import Any

import tensorflow as tf

from .efficientnet import EfficientNetParams, EfficientNetV2
from .multiresnet import MultiresNet, MultiresNetParams
from .resnet import ResNet, ResNetParams
from .tcn import Tcn, TcnParams
from .unet import UNet, UNetParams
from .unext import UNext, UNextParams


def generate_model(
    inputs: tf.Tensor,
    num_classes: int,
    name: str,
    params: dict[str, Any],
) -> tf.keras.Model:
    """Model factory: Generates a model based on the provided name and parameters

    Args:
        inputs (tf.Tensor): Input tensor
        num_classes (int): Number of classes
        name (str): Model name
        params (dict[str, Any]): Model parameters

    Returns:
        tf.keras.Model: Generated model
    """
    if params is None:
        raise ValueError("Model parameters must be provided")

    match name:
        case "unet":
            return UNet(x=inputs, params=UNetParams.parse_obj(params), num_classes=num_classes)

        case "unext":
            return UNext(x=inputs, params=UNextParams.parse_obj(params), num_classes=num_classes)

        case "resnet":
            return ResNet(x=inputs, params=ResNetParams.parse_obj(params), num_classes=num_classes)

        case "multiresnet":
            return MultiresNet(
                x=inputs,
                params=MultiresNetParams.parse_obj(params),
                num_classes=num_classes,
            )

        case "efficientnetv2":
            return EfficientNetV2(x=inputs, params=EfficientNetParams.parse_obj(params), num_classes=num_classes)

        case "tcn":
            return Tcn(x=inputs, params=TcnParams.parse_obj(params), num_classes=num_classes)

        case _:
            raise NotImplementedError()
    # END MATCH
