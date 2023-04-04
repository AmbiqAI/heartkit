""" EfficientNet https://arxiv.org/abs/2104.00298"""
from typing import Callable

import tensorflow as tf
from keras.engine.keras_tensor import KerasTensor
from pydantic import BaseModel, Field

from .blocks import batch_norm, conv2d, make_divisible, mbconv_block, relu6


class MBConvParams(BaseModel):
    """MBConv parameters"""

    filters: int = Field(..., description="# filters")
    depth: int = Field(default=1, description="Layer depth")
    ex_ratio: float = Field(default=1, description="Expansion ratio")
    kernel_size: int | tuple[int, int] = Field(default=3, description="Kernel size")
    strides: int | tuple[int, int] = Field(default=1, description="Stride size")
    se_ratio: float = Field(default=8, description="Squeeze Excite ratio")
    droprate: float = Field(default=0, description="Drop rate")


class EfficientNetParams(BaseModel):
    """EfficientNet parameters"""

    blocks: list[MBConvParams] = Field(
        default_factory=list, description="EfficientNet blocks"
    )
    input_filters: int = Field(default=0, description="Input filters")
    input_kernel_size: int | tuple[int, int] = Field(
        default=3, description="Input kernel size"
    )
    input_strides: int | tuple[int, int] = Field(default=2, description="Input stride")
    output_filters: int = Field(default=0, description="Output filters")
    include_top: bool = Field(default=True, description="Include top")
    dropout: float = Field(default=0.2, description="Dropout rate")
    drop_connect_rate: float = Field(default=0.2, description="Drop connect rate")
    model_name: str = Field(default="EfficientNetV2", description="Model name")


def efficientnet_core(
    blocks: list[MBConvParams], drop_connect_rate: float = 0
) -> Callable[[KerasTensor], KerasTensor]:
    """EfficientNet core

    Args:
        blocks (list[MBConvParam]): MBConv params
        drop_connect_rate (float, optional): Drop connect rate. Defaults to 0.

    Returns:
        Callable[[KerasTensor], KerasTensor]: Core
    """

    def layer(x: KerasTensor) -> KerasTensor:
        global_block_id = 0
        total_blocks = sum((b.depth for b in blocks))
        for i, block in enumerate(blocks):
            filters = make_divisible(block.filters, 8)
            for d in range(block.depth):
                name = f"stage{i+1}.mbconv{d+1}"
                block_drop_rate = drop_connect_rate * global_block_id / total_blocks
                x = mbconv_block(
                    filters,
                    block.ex_ratio,
                    block.kernel_size,
                    block.strides if d == 0 else 1,
                    block.se_ratio,
                    droprate=block_drop_rate,
                    name=name,
                )(x)
                global_block_id += 1
            # END FOR
        # END FOR
        return x

    return layer


def EfficientNetV2(
    x: KerasTensor,
    params: EfficientNetParams,
    num_classes: int | None = None,
):
    """Create EfficientNet V2 TF functional model

    Args:
        x (KerasTensor): Input tensor
        params (EfficientNetParams): Model parameters.
        num_classes (int, optional): # classes.

    Returns:
        tf.keras.Model: Model
    """
    # Stem
    if params.input_filters > 0:
        name = "stem"
        filters = make_divisible(params.input_filters, 8)
        y = conv2d(
            filters,
            kernel_size=params.input_kernel_size,
            strides=params.input_strides,
            name=name,
        )(x)
        y = batch_norm(name=name)(y)
        y = relu6(name=name)(y)
    else:
        y = x

    y = efficientnet_core(
        blocks=params.blocks, drop_connect_rate=params.drop_connect_rate
    )(y)

    if params.output_filters:
        name = "neck"
        filters = make_divisible(params.output_filters, 8)
        y = conv2d(
            filters, kernel_size=(1, 1), strides=(1, 1), padding="same", name=name
        )(y)
        y = batch_norm(name=name)(y)
        y = relu6(name=name)(y)

    if params.include_top:
        name = "top"
        y = tf.keras.layers.GlobalAveragePooling2D(name=f"{name}.pool")(y)
        if 0 < params.dropout < 1:
            y = tf.keras.layers.Dropout(params.dropout)(y)
        y = tf.keras.layers.Dense(num_classes, name=name)(y)
    model = tf.keras.Model(x, y, name=params.model_name)
    return model
