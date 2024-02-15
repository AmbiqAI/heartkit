""" ResNet """

import keras
import tensorflow as tf
from pydantic import BaseModel, Field

from .blocks import batch_norm, conv2d, relu6
from .defines import KerasLayer


class ResNetBlockParams(BaseModel):
    """ResNet block parameters"""

    filters: int = Field(..., description="# filters")
    depth: int = Field(default=1, description="Layer depth")
    kernel_size: int | tuple[int, int] = Field(default=3, description="Kernel size")
    strides: int | tuple[int, int] = Field(default=1, description="Stride size")
    bottleneck: bool = Field(default=False, description="Use bottleneck blocks")


class ResNetParams(BaseModel):
    """ResNet parameters"""

    blocks: list[ResNetBlockParams] = Field(default_factory=list, description="ResNet blocks")
    input_filters: int = Field(default=0, description="Input filters")
    input_kernel_size: int | tuple[int, int] = Field(default=3, description="Input kernel size")
    input_strides: int | tuple[int, int] = Field(default=2, description="Input stride")
    include_top: bool = Field(default=True, description="Include top")
    dropout: float = Field(default=0.2, description="Dropout rate")
    model_name: str = Field(default="ResNet", description="Model name")


def generate_bottleneck_block(
    filters: int,
    kernel_size: int | tuple[int, int] = 3,
    strides: int | tuple[int, int] = 1,
    expansion: int = 4,
) -> KerasLayer:
    """Generate functional bottleneck block.

    Args:
        filters (int): Filter size
        kernel_size (int | tuple[int, int], optional): Kernel size. Defaults to 3.
        strides (int | tuple[int, int], optional): Stride length. Defaults to 1.
        expansion (int, optional): Expansion factor. Defaults to 4.

    Returns:
        KerasLayer: TF functional layer
    """

    def layer(x: tf.Tensor) -> tf.Tensor:
        num_chan = x.shape[-1]
        projection = num_chan != filters * expansion or (strides > 1 if isinstance(strides, int) else strides[0] > 1)

        bx = conv2d(filters, 1, 1)(x)
        bx = batch_norm()(bx)
        bx = relu6()(bx)

        bx = conv2d(filters, kernel_size, strides)(x)
        bx = batch_norm()(bx)
        bx = relu6()(bx)

        bx = conv2d(filters * expansion, 1, 1)(bx)
        bx = batch_norm()(bx)

        if projection:
            x = conv2d(filters * expansion, 1, strides)(x)
            x = batch_norm()(x)
        x = keras.layers.Add()([bx, x])
        x = relu6()(x)
        return x

    return layer


def generate_residual_block(
    filters: int,
    kernel_size: int | tuple[int, int] = 3,
    strides: int | tuple[int, int] = 1,
) -> KerasLayer:
    """Generate functional residual block

    Args:
        filters (int): Filter size
        kernel_size (int | tuple[int, int], optional): Kernel size. Defaults to 3.
        strides (int | tuple[int, int], optional): Stride length. Defaults to 1.

    Returns:
        KerasLayer: TF functional layer
    """

    def layer(x: tf.Tensor) -> tf.Tensor:
        num_chan = x.shape[-1]
        projection = num_chan != filters or (strides > 1 if isinstance(strides, int) else strides[0] > 1)
        bx = conv2d(filters, kernel_size, strides)(x)
        bx = batch_norm()(bx)
        bx = relu6()(bx)

        bx = conv2d(filters, kernel_size, 1)(bx)
        bx = batch_norm()(bx)
        if projection:
            x = conv2d(filters, 1, strides)(x)
            x = batch_norm()(x)
        x = keras.layers.Add()([bx, x])
        x = relu6()(x)
        return x

    return layer


def ResNet(
    x: tf.Tensor,
    params: ResNetParams,
    num_classes: int | None = None,
) -> keras.Model:
    """Generate functional ResNet model.
    Args:
        x (tf.Tensor): Inputs
        params (ResNetParams): Model parameters.
        num_classes (int, optional): # class outputs. Defaults to None.

    Returns:
        keras.Model: Model
    """

    requires_reshape = len(x.shape) == 3
    if requires_reshape:
        y = keras.layers.Reshape((1,) + x.shape[1:])(x)
    else:
        y = x
    # END IF

    if params.input_filters:
        y = conv2d(
            params.input_filters,
            kernel_size=params.input_kernel_size,
            strides=params.input_strides,
        )(y)
        y = batch_norm()(y)
        y = relu6()(y)
    # END IF

    for stage, block in enumerate(params.blocks):
        for d in range(block.depth):
            func = generate_bottleneck_block if block.bottleneck else generate_residual_block
            y = func(
                filters=block.filters,
                kernel_size=block.kernel_size,
                strides=block.strides if d == 0 and stage > 0 else 1,
            )(y)
        # END FOR
    # END FOR

    if params.include_top:
        y = keras.layers.GlobalAveragePooling2D()(y)
        y = keras.layers.Dense(num_classes)(y)

    model = keras.Model(x, y, name="model")
    return model
