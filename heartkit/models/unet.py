""" UNet """

import tensorflow as tf
from pydantic import BaseModel, Field

from .blocks import batch_norm, conv2d, relu6


class UNetBlockParams(BaseModel):
    """UNet block parameters"""

    filters: int = Field(..., description="# filters")
    depth: int = Field(default=1, description="Layer depth")
    kernel: int | tuple[int, int] = Field(default=3, description="Kernel size")
    pool: int | tuple[int, int] = Field(default=3, description="Pool size")
    strides: int | tuple[int, int] = Field(default=1, description="Stride size")
    skip: bool = Field(default=True, description="Add skip connection")


class UNetParams(BaseModel):
    """UNet parameters"""

    blocks: list[UNetBlockParams] = Field(default_factory=list, description="UNet blocks")
    include_top: bool = Field(default=True, description="Include top")
    model_name: str = Field(default="UNet", description="Model name")
    output_kernel_size: int | tuple[int, int] = Field(default=3, description="Output kernel size")


def UNet(
    x: tf.Tensor,
    params: UNetParams,
    num_classes: int,
) -> tf.keras.Model:
    """Create UNet TF functional model

    Args:
        x (tf.Tensor): Input tensor
        params (ResNetParams): Model parameters.
        num_classes (int, optional): # classes.

    Returns:
        tf.keras.Model: Model
    """
    y = x
    skip_layers: list[tf.keras.layers.Layer | None] = []
    for i, block in enumerate(params.blocks):
        name = f"ENC{i+1}"

        ym = y
        for d in range(block.depth):
            ym = tf.keras.layers.SeparableConv2D(
                block.filters,
                kernel_size=block.kernel,
                strides=(1, 1),
                padding="same",
                depthwise_initializer="he_normal",
                pointwise_initializer="he_normal",
                name=f"{name}.CONV{d+1}",
            )(ym)
            ym = batch_norm(name=f"{name}.BN{d+1}")(ym)
            ym = relu6(name=f"{name}.ACT{d+1}")(ym)
        # END FOR

        # Project residual
        yr = conv2d(
            block.filters,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding="same",
            name=f"{name}.CONV_RES",
        )(y)
        y = tf.keras.layers.add([ym, yr], name=f"{name}.ADD1")

        skip_layers.append(y if block.skip else None)

        y = tf.keras.layers.MaxPooling2D(block.pool, strides=block.strides, padding="same", name=f"{name}.POOL1")(y)
    # END FOR

    for i, block in enumerate(reversed(params.blocks)):
        name = f"DEC{i+1}"

        # Expand inputs via transpose w/ stride
        y = tf.keras.layers.Conv2DTranspose(
            block.filters,
            kernel_size=block.kernel,
            strides=block.strides,
            padding="same",
            kernel_initializer="he_normal",
            name=f"{name}.CONV1",
        )(y)
        y = batch_norm(name=f"{name}.BN1")(y)
        y = relu6(name=f"{name}.ACT1")(y)

        # Add skip connection
        skip_layer = skip_layers.pop()
        if skip_layer is not None:
            y = tf.keras.layers.concatenate([y, skip_layer], name=f"{name}.CAT1")  # Can add or concatenate

            # Use 1x1 conv to reduce filters
            y = conv2d(
                block.filters,
                kernel_size=(1, 1),
                padding="same",
                name=f"{name}.CONV2",
            )(y)
            y = batch_norm(name=f"{name}.BN2")(y)
            y = relu6(name=f"{name}.ACT2")(y)

        ym = tf.keras.layers.SeparableConv2D(
            block.filters,
            kernel_size=block.kernel,
            strides=(1, 1),
            padding="same",
            depthwise_initializer="he_normal",
            pointwise_initializer="he_normal",
            name=f"{name}.CONV3",
        )(y)
        ym = batch_norm(name=f"{name}.BN3")(ym)
        ym = relu6(name=f"{name}.ACT3")(ym)

        # Project residual
        yr = conv2d(block.filters, (1, 1), name=f"{name}.CONV_RES")(y)
        y = tf.keras.layers.add([ym, yr], name=f"{name}.ADD1")  # Add back residual
    # END FOR

    if params.include_top:
        # Add a per-point classification layer
        y = conv2d(num_classes, params.output_kernel_size, name="NECK.CONV1")(y)
        y = tf.keras.layers.Reshape(y.shape[2:])(y)

    # Define the model
    model = tf.keras.Model(x, y, name=params.model_name)
    return model
