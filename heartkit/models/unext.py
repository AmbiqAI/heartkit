""" UNext """

from typing import Literal

import keras
import tensorflow as tf
from pydantic import BaseModel, Field

from .defines import KerasLayer


class UNextBlockParams(BaseModel):
    """UNext block parameters"""

    filters: int = Field(..., description="# filters")
    depth: int = Field(default=1, description="Layer depth")
    ddepth: int | None = Field(default=None, description="Layer decoder depth")
    kernel: int | tuple[int, int] = Field(default=3, description="Kernel size")
    pool: int | tuple[int, int] = Field(default=2, description="Pool size")
    strides: int | tuple[int, int] = Field(default=2, description="Stride size")
    skip: bool = Field(default=True, description="Add skip connection")
    expand_ratio: float = Field(default=1, description="Expansion ratio")
    se_ratio: float = Field(default=0, description="Squeeze and excite ratio")
    dropout: float | None = Field(default=None, description="Dropout rate")
    norm: Literal["batch", "layer"] | None = Field(default="layer", description="Normalization type")


class UNextParams(BaseModel):
    """UNext parameters"""

    blocks: list[UNextBlockParams] = Field(default_factory=list, description="UNext blocks")
    include_top: bool = Field(default=True, description="Include top")
    use_logits: bool = Field(default=True, description="Use logits")
    model_name: str = Field(default="UNext", description="Model name")
    output_kernel_size: int | tuple[int, int] = Field(default=3, description="Output kernel size")
    output_kernel_stride: int | tuple[int, int] = Field(default=1, description="Output kernel stride")


def se_block(ratio: int = 8, name: str | None = None):
    """Squeeze and excite block"""

    def layer(x: tf.Tensor) -> tf.Tensor:
        num_chan = x.shape[-1]
        # Squeeze
        y = keras.layers.GlobalAveragePooling2D(name=f"{name}.pool" if name else None, keepdims=True)(x)

        y = keras.layers.Conv2D(num_chan // ratio, kernel_size=1, use_bias=True, name=f"{name}.sq" if name else None)(y)

        y = keras.layers.Activation(tf.nn.relu6, name=f"{name}.relu" if name else None)(y)

        # Excite
        y = keras.layers.Conv2D(num_chan, kernel_size=1, use_bias=True, name=f"{name}.ex" if name else None)(y)
        y = keras.layers.Activation(keras.activations.hard_sigmoid, name=f"{name}.sigg" if name else None)(y)
        y = keras.layers.Multiply(name=f"{name}.mul" if name else None)([x, y])
        return y

    return layer


def norm_layer(norm: str, name: str) -> KerasLayer:
    """Normalization layer

    Args:
        norm (str): Normalization type
        name (str): Name

    Returns:
        KerasLayer: Layer
    """

    def layer(x: tf.Tensor) -> tf.Tensor:
        """Functional normalization layer

        Args:
            x (tf.Tensor): Input tensor

        Returns:
            tf.Tensor: Output tensor
        """
        if norm == "batch":
            return keras.layers.BatchNormalization(axis=-1, name=f"{name}.BN")(x)
        if norm == "layer":
            ln_axis = 2 if x.shape[1] == 1 else 1 if x.shape[2] == 1 else (1, 2)
            return keras.layers.LayerNormalization(axis=ln_axis, name=f"{name}.LN")(x)
        return x

    return layer


def UNext_block(
    output_filters: int,
    expand_ratio: float = 1,
    kernel_size: int | tuple[int, int] = 3,
    strides: int | tuple[int, int] = 1,
    se_ratio: float = 4,
    dropout: float | None = 0,
    norm: Literal["batch", "layer"] | None = "batch",
    name: str = "",
) -> KerasLayer:
    """Create UNext block"""

    def layer(x: tf.Tensor) -> tf.Tensor:
        input_filters: int = x.shape[-1]
        strides_len = strides if isinstance(strides, int) else sum(strides) // len(strides)
        add_residual = input_filters == output_filters and strides_len == 1

        # Inverted expansion block
        if expand_ratio != 1:
            y = keras.layers.Conv2D(
                filters=int(expand_ratio * input_filters),
                kernel_size=1,
                strides=1,
                padding="same",
                use_bias=norm is None,
                kernel_initializer="he_normal",
                kernel_regularizer=keras.regularizers.L2(1e-3),
                name=f"{name}.EX",
            )(y)
            norm_layer(norm, f"{name}.EX")(y)
            y = keras.layers.Activation(tf.nn.relu6, name=f"{name}.EX.ACT")(y)
        # END IF

        # Depthwise conv
        y = keras.layers.DepthwiseConv2D(
            kernel_size=kernel_size,
            strides=1,
            padding="same",
            use_bias=norm is None,
            depthwise_initializer="he_normal",
            depthwise_regularizer=keras.regularizers.L2(1e-3),
            name=f"{name}.DW",
        )(x)
        norm_layer(norm, f"{name}.DW")(y)
        y = keras.layers.Activation(tf.nn.relu6, name=f"{name}.DW.ACT")(y)

        # Squeeze and excite
        if se_ratio > 1:
            y = se_block(ratio=se_ratio, name=f"{name}.SE")(y)

        y = keras.layers.Conv2D(
            filters=output_filters,
            kernel_size=1,
            strides=1,
            padding="same",
            use_bias=norm is None,
            kernel_initializer="he_normal",
            kernel_regularizer=keras.regularizers.L2(1e-3),
            name=f"{name}.RD",
        )(y)
        norm_layer(norm, f"{name}.RD")(y)

        if add_residual:
            if dropout and dropout > 0:
                y = keras.layers.Dropout(
                    dropout,
                    noise_shape=(y.shape),
                    name=f"{name}.DO" if name else None,
                )(y)
            y = keras.layers.Add(name=f"{name}.RS" if name else None)([x, y])
        return y

    # END DEF
    return layer


def unext_core(
    x: tf.Tensor,
    params: UNextParams,
) -> tf.Tensor:
    """Create UNext TF functional core

    Args:
        x (tf.Tensor): Input tensor
        params (UNextParams): Model parameters.

    Returns:
        tf.Tensor: Output tensor
    """

    y = x

    #### ENCODER ####
    skip_layers: list[keras.layers.Layer | None] = []
    for i, block in enumerate(params.blocks):
        name = f"ENC{i+1}"
        for d in range(block.depth):
            y = UNext_block(
                output_filters=block.filters,
                expand_ratio=block.expand_ratio,
                kernel_size=block.kernel,
                strides=1,
                se_ratio=block.se_ratio,
                dropout=block.dropout,
                norm=block.norm,
                name=f"{name}.D{d+1}",
            )(y)
        # END FOR
        skip_layers.append(y if block.skip else None)

        y = keras.layers.MaxPooling2D(block.pool, strides=block.strides, padding="same", name=f"{name}.POOL")(y)

        # # Downsample using strided conv
        # y = keras.layers.Conv2D(
        #     filters=block.filters,
        #     kernel_size=block.pool,
        #     strides=block.strides,
        #     padding="same",
        #     use_bias=block.norm is None,
        #     kernel_initializer="he_normal",
        #     kernel_regularizer=keras.regularizers.L2(1e-3),
        #     name=f"{name}.PL",
        # )(y)
        # norm_layer(block.norm, f"{name}.RD")(y)
    # END FOR

    #### DECODER ####
    for i, block in enumerate(reversed(params.blocks)):
        name = f"DEC{i+1}"
        for d in range(block.ddepth or block.depth):
            y = UNext_block(
                output_filters=block.filters,
                expand_ratio=block.expand_ratio,
                kernel_size=block.kernel,
                strides=1,
                se_ratio=block.se_ratio,
                dropout=block.dropout,
                norm=block.norm,
                name=f"{name}.D{d+1}",
            )(y)
        # END FOR

        # Upsample using transposed conv
        # y = keras.layers.Conv1DTranspose(
        #     filters=block.filters,
        #     kernel_size=block.pool,
        #     strides=block.strides,
        #     padding="same",
        #     kernel_initializer="he_normal",
        #     kernel_regularizer=keras.regularizers.L2(1e-3),
        #     name=f"{name}.unpool",
        # )(y)

        # y = keras.layers.Conv2D(
        #     filters=block.filters,
        #     kernel_size=block.pool,
        #     strides=1,
        #     padding="same",
        #     use_bias=block.norm is None,
        #     kernel_initializer="he_normal",
        #     kernel_regularizer=keras.regularizers.L2(1e-3),
        #     name=f"{name}.conv",
        # )(y)
        y = keras.layers.UpSampling2D(size=block.strides, name=f"{name}.POOL")(y)

        # Skip connection
        skip_layer = skip_layers.pop()
        if skip_layer is not None:
            y = keras.layers.Concatenate(name=f"{name}.SL.CAT")([y, skip_layer])
            # y = keras.layers.Add(name=f"{name}.S1.cat")([y, skip_layer])
            # Use 1x1 conv to reduce filters
            y = keras.layers.Conv2D(
                block.filters,
                kernel_size=1,  # block.kernel,
                padding="same",
                kernel_initializer="he_normal",
                kernel_regularizer=keras.regularizers.L2(1e-3),
                use_bias=block.norm is None,
                name=f"{name}.SL.CONV",
            )(y)
            norm_layer(block.norm, f"{name}.SL")(y)
            y = keras.layers.Activation(
                tf.nn.relu6,
                name=f"{name}.SL.ACT",
            )(y)
        # END IF

        y = UNext_block(
            output_filters=block.filters,
            expand_ratio=block.expand_ratio,
            kernel_size=block.kernel,
            strides=1,
            se_ratio=block.se_ratio,
            dropout=block.dropout,
            norm=block.norm,
            name=f"{name}.D{block.depth+1}",
        )(y)

    # END FOR
    return y


def UNext(
    x: tf.Tensor,
    params: UNextParams,
    num_classes: int,
) -> keras.Model:
    """Create UNext TF functional model

    Args:
        x (tf.Tensor): Input tensor
        params (UNextParams): Model parameters.
        num_classes (int, optional): # classes.

    Returns:
        keras.Model: Model
    """
    requires_reshape = len(x.shape) == 3
    if requires_reshape:
        y = keras.layers.Reshape((1,) + x.shape[1:])(x)
    else:
        y = x

    y = unext_core(y, params)

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

    # Define the model
    model = keras.Model(x, y, name=params.model_name)
    return model


def unext_from_object(
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
    return UNext(x=x, params=UNextParams(**params), num_classes=num_classes)
