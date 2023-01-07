from typing import Optional

import tensorflow as tf
from keras.engine.keras_tensor import KerasTensor

from ..types import ArchitectureType
from .resnet1d import generate_resnet


def ecg_feature_extractor(
    inputs: KerasTensor,
    arch: Optional[ArchitectureType] = None,
    stages: Optional[int] = None,
) -> KerasTensor:
    """AI based feature extractor. Currently consists of 1D variant of ResNet

    Args:
        arch (str, optional): Architecture name. Defaults to None.
        stages (int, optional): Number of stages. Defaults to None.

    Returns:
        tf.keras.Sequential: Feature extractor model
    """
    if arch is None or arch == "resnet12":
        x = generate_resnet(
            inputs=inputs,
            input_conv=(32, 7, 2),
            blocks=(1, 1, 1)[:stages],
            filters=(32, 64, 128),
            kernel_size=(7, 5, 3),
            include_top=False,
        )
    elif arch == "resnet18":
        x = generate_resnet(
            inputs=inputs,
            blocks=(2, 2, 2, 2)[:stages],
            kernel_size=(7, 5, 5, 3),
            include_top=False,
        )
    elif arch == "resnet34":
        x = generate_resnet(
            inputs=inputs,
            blocks=(3, 4, 6, 3)[:stages],
            kernel_size=(7, 5, 5, 3),
            include_top=False,
        )
    elif arch == "resnet50":
        x = generate_resnet(
            inputs=inputs,
            blocks=(3, 4, 6, 3)[:stages],
            kernel_size=(7, 5, 5, 3),
            use_bottleneck=True,
            include_top=False,
        )
    else:
        raise ValueError(f"unknown architecture: {arch}")

    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    return x
