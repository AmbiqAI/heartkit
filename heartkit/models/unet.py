""" UNet """

from typing import Literal

import keras
import tensorflow as tf
from pydantic import BaseModel, Field

from .blocks import batch_norm, layer_norm, relu6


class UNetBlockParams(BaseModel):
    """UNet block parameters"""

    filters: int = Field(..., description="# filters")
    depth: int = Field(default=1, description="Layer depth")
    ddepth: int | None = Field(default=None, description="Decoder depth")
    kernel: int | tuple[int, int] = Field(default=3, description="Kernel size")
    pool: int | tuple[int, int] = Field(default=3, description="Pool size")
    strides: int | tuple[int, int] = Field(default=1, description="Stride size")
    skip: bool = Field(default=True, description="Add skip connection")
    seperable: bool = Field(default=False, description="Use seperable convs")
    dropout: float | None = Field(default=None, description="Dropout rate")
    norm: Literal["batch", "layer"] | None = Field(default="batch", description="Normalization type")
    dilation: int | tuple[int, int] | None = Field(default=None, description="Dilation factor")


class UNetParams(BaseModel):
    """UNet parameters"""

    blocks: list[UNetBlockParams] = Field(default_factory=list, description="UNet blocks")
    include_top: bool = Field(default=True, description="Include top")
    use_logits: bool = Field(default=True, description="Use logits")
    model_name: str = Field(default="UNet", description="Model name")
    output_kernel_size: int | tuple[int, int] = Field(default=3, description="Output kernel size")
    output_kernel_stride: int | tuple[int, int] = Field(default=1, description="Output kernel stride")


def UNet(
    x: tf.Tensor,
    params: UNetParams,
    num_classes: int,
) -> keras.Model:
    """Create UNet TF functional model

    Args:
        x (tf.Tensor): Input tensor
        params (ResNetParams): Model parameters.
        num_classes (int, optional): # classes.

    Returns:
        keras.Model: Model
    """
    requires_reshape = len(x.shape) == 3
    if requires_reshape:
        y = keras.layers.Reshape((1,) + x.shape[1:])(x)
    else:
        y = x
    # END IF

    #### ENCODER ####
    skip_layers: list[keras.layers.Layer | None] = []
    for i, block in enumerate(params.blocks):
        name = f"ENC{i+1}"
        ym = y
        for d in range(block.depth):
            dname = f"{name}.D{d+1}"
            if block.dilation is None:
                dilation_rate = (1, 1)
            elif isinstance(block.dilation, int):
                dilation_rate = (block.dilation**d, block.dilation**d)
            else:
                dilation_rate = (block.dilation[0] ** d, block.dilation[1] ** d)
            if block.seperable:
                ym = keras.layers.SeparableConv2D(
                    block.filters,
                    kernel_size=block.kernel,
                    strides=(1, 1),
                    padding="same",
                    dilation_rate=dilation_rate,
                    depthwise_initializer="he_normal",
                    pointwise_initializer="he_normal",
                    depthwise_regularizer=keras.regularizers.L2(1e-3),
                    pointwise_regularizer=keras.regularizers.L2(1e-3),
                    use_bias=block.norm is None,
                    name=f"{dname}.conv",
                )(ym)
            else:
                ym = keras.layers.Conv2D(
                    block.filters,
                    kernel_size=block.kernel,
                    strides=(1, 1),
                    padding="same",
                    dilation_rate=dilation_rate,
                    kernel_initializer="he_normal",
                    kernel_regularizer=keras.regularizers.L2(1e-3),
                    use_bias=block.norm is None,
                    name=f"{dname}.conv",
                )(ym)
            if block.norm == "layer":
                ym = layer_norm(name=dname, axis=[1, 2])(ym)
            elif block.norm == "batch":
                ym = batch_norm(name=dname, momentum=0.99)(ym)
            ym = relu6(name=dname)(ym)
        # END FOR

        # Project residual
        yr = keras.layers.Conv2D(
            block.filters,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding="same",
            kernel_initializer="he_normal",
            kernel_regularizer=keras.regularizers.L2(1e-3),
            name=f"{name}.skip",
        )(y)

        if block.dropout is not None:
            ym = keras.layers.Dropout(block.dropout, noise_shape=ym.shape)(ym)
        y = keras.layers.add([ym, yr], name=f"{name}.add")

        skip_layers.append(y if block.skip else None)

        y = keras.layers.MaxPooling2D(block.pool, strides=block.strides, padding="same", name=f"{name}.pool")(y)
    # END FOR

    #### DECODER ####
    for i, block in enumerate(reversed(params.blocks)):
        name = f"DEC{i+1}"
        for d in range(block.ddepth or block.depth):
            dname = f"{name}.D{d+1}"
            if block.seperable:
                y = keras.layers.SeparableConv2D(
                    block.filters,
                    kernel_size=block.kernel,
                    strides=(1, 1),
                    padding="same",
                    dilation_rate=dilation_rate,
                    depthwise_initializer="he_normal",
                    pointwise_initializer="he_normal",
                    depthwise_regularizer=keras.regularizers.L2(1e-3),
                    pointwise_regularizer=keras.regularizers.L2(1e-3),
                    use_bias=block.norm is None,
                    name=f"{dname}.conv",
                )(y)
            else:
                y = keras.layers.Conv2D(
                    block.filters,
                    kernel_size=block.kernel,
                    strides=(1, 1),
                    padding="same",
                    dilation_rate=dilation_rate,
                    kernel_initializer="he_normal",
                    kernel_regularizer=keras.regularizers.L2(1e-3),
                    use_bias=block.norm is None,
                    name=f"{dname}.conv",
                )(y)
            if block.norm == "layer":
                y = layer_norm(name=dname, axis=[1, 2])(y)
            elif block.norm == "batch":
                y = batch_norm(name=dname, momentum=0.99)(y)
            y = relu6(name=dname)(y)
        # END FOR

        y = keras.layers.UpSampling2D(size=block.strides, name=f"{dname}.unpool")(y)

        # Add skip connection
        dname = f"{name}.D{block.depth+1}"
        skip_layer = skip_layers.pop()
        if skip_layer is not None:
            y = keras.layers.concatenate([y, skip_layer], name=f"{dname}.cat")  # Can add or concatenate
            # Use 1x1 conv to reduce filters
            y = keras.layers.Conv2D(
                block.filters,
                kernel_size=(1, 1),
                padding="same",
                kernel_initializer="he_normal",
                kernel_regularizer=keras.regularizers.L2(1e-3),
                use_bias=block.norm is None,
                name=f"{dname}.conv",
            )(y)
            if block.norm == "layer":
                y = layer_norm(name=dname, axis=[1, 2])(y)
            elif block.norm == "batch":
                y = batch_norm(name=dname, momentum=0.99)(y)
            y = relu6(name=dname)(y)
        # END IF

        dname = f"{name}.D{block.depth+2}"
        if block.seperable:
            ym = keras.layers.SeparableConv2D(
                block.filters,
                kernel_size=block.kernel,
                strides=(1, 1),
                padding="same",
                depthwise_initializer="he_normal",
                pointwise_initializer="he_normal",
                depthwise_regularizer=keras.regularizers.L2(1e-3),
                pointwise_regularizer=keras.regularizers.L2(1e-3),
                use_bias=block.norm is None,
                name=f"{dname}.conv",
            )(y)
        else:
            ym = keras.layers.Conv2D(
                block.filters,
                kernel_size=block.kernel,
                strides=(1, 1),
                padding="same",
                kernel_initializer="he_normal",
                kernel_regularizer=keras.regularizers.L2(1e-3),
                use_bias=block.norm is None,
                name=f"{dname}.conv",
            )(y)
        if block.norm == "layer":
            ym = layer_norm(name=dname, axis=[1, 2])(ym)
        elif block.norm == "batch":
            ym = batch_norm(name=dname, momentum=0.99)(ym)
        ym = relu6(name=dname)(ym)

        # Project residual
        yr = keras.layers.Conv2D(
            block.filters,
            kernel_size=(1, 1),
            padding="same",
            kernel_initializer="he_normal",
            kernel_regularizer=keras.regularizers.L2(1e-3),
            name=f"{name}.skip",
        )(y)
        y = keras.layers.add([ym, yr], name=f"{name}.add")  # Add back residual
    # END FOR

    if params.include_top:
        # Add a per-point classification layer
        y = keras.layers.Conv2D(
            num_classes,
            kernel_size=params.output_kernel_size,
            padding="same",
            kernel_initializer="he_normal",
            kernel_regularizer=keras.regularizers.L2(1e-3),
            name="NECK.conv",
            use_bias=True,
        )(y)
        if not params.use_logits:
            y = keras.layers.Softmax()(y)
        # END IF
    # END IF

    if requires_reshape:
        y = keras.layers.Reshape(y.shape[2:])(y)
    # END IF
    # Define the model
    model = keras.Model(x, y, name=params.model_name)
    return model


def unet_from_object(
    x: tf.Tensor,
    params: dict,
    num_classes: int,
) -> keras.Model:
    """Create model from object

    Args:
        x (tf.Tensor): Input tensor
        params (dict): Model parameters.
        num_classes (int, optional): # classes.

    Returns:
        keras.Model: Model
    """
    return UNet(x=x, params=UNetParams(**params), num_classes=num_classes)
