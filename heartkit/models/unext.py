""" UNext """
from typing import Literal

import tensorflow as tf
from pydantic import BaseModel, Field


class UNextBlockParams(BaseModel):
    """UNext block parameters"""

    filters: int = Field(..., description="# filters")
    depth: int = Field(default=1, description="Layer depth")
    ddepth: int | None = Field(default=None, description="Decoder depth")
    kernel: int = Field(default=3, description="Kernel size")
    pool: int = Field(default=2, description="Pool size")
    strides: int = Field(default=2, description="Stride size")
    skip: bool = Field(default=True, description="Add skip connection")
    expand_ratio: float = Field(default=1, description="Expansion ratio")
    se_ratio: float = Field(default=0, description="Squeeze and excite ratio")
    dropout: float | None = Field(default=None, description="Dropout rate")
    norm: Literal["batch", "layer"] | None = Field(default="batch", description="Normalization type")


class UNextParams(BaseModel):
    """UNext parameters"""

    blocks: list[UNextBlockParams] = Field(default_factory=list, description="UNext blocks")
    include_top: bool = Field(default=True, description="Include top")
    use_logits: bool = Field(default=True, description="Use logits")
    model_name: str = Field(default="UNext", description="Model name")
    output_kernel_size: int = Field(default=3, description="Output kernel size")
    output_kernel_stride: int = Field(default=1, description="Output kernel stride")


def se_block(ratio: int = 8, name: str | None = None):
    """Squeeze and excite block"""

    def layer(x: tf.Tensor) -> tf.Tensor:
        num_chan = x.shape[-1]
        # Squeeze
        y = tf.keras.layers.GlobalAveragePooling1D(name=f"{name}.pool" if name else None, keepdims=True)(x)

        y = tf.keras.layers.Conv1D(
            num_chan // ratio, kernel_size=1, use_bias=True, name=f"{name}.sq" if name else None
        )(y)

        y = tf.keras.layers.Activation(tf.nn.relu6, name=f"{name}.relu" if name else None)(y)

        # Excite
        y = tf.keras.layers.Conv1D(num_chan, kernel_size=1, use_bias=True, name=f"{name}.ex" if name else None)(y)
        y = tf.keras.layers.Activation(tf.keras.activations.hard_sigmoid, name=f"{name}.sigg" if name else None)(y)
        y = tf.keras.layers.Multiply(name=f"{name}.mul" if name else None)([x, y])
        return y

    return layer


def UNext_block(
    output_filters: int,
    expand_ratio: float = 1,
    kernel_size: int = 3,
    strides: int = 1,
    se_ratio: float = 4,
    droprate: float | None = 0,
    name: str | None = None,
):
    """Create UNext block"""

    def layer(x: tf.Tensor) -> tf.Tensor:
        input_filters: int = x.shape[-1]
        add_residual = input_filters == output_filters and strides == 1

        # Depthwise conv
        y = tf.keras.layers.DepthwiseConv1D(
            kernel_size=kernel_size,
            strides=1,
            padding="same",
            depthwise_initializer="he_normal",
            depthwise_regularizer=tf.keras.regularizers.L2(1e-3),
            name=f"{name}.dwconv" if name else None,
        )(x)
        y = tf.keras.layers.LayerNormalization(
            axis=(1),
            name=f"{name}.norm" if name else None,
        )(y)

        # Inverted expansion block (use Pointwise?)
        y = tf.keras.layers.Conv1D(
            filters=int(expand_ratio * input_filters),
            kernel_size=1,
            strides=1,
            padding="same",
            groups=input_filters,
            kernel_initializer="he_normal",
            kernel_regularizer=tf.keras.regularizers.L2(1e-3),
            name=f"{name}.expand" if name else None,
        )(y)

        y = tf.keras.layers.Activation(
            tf.nn.relu6,
            name=f"{name}.relu" if name else None,
        )(y)

        # Squeeze and excite
        if se_ratio > 1:
            name_se = f"{name}.se" if name else None
            y = se_block(ratio=se_ratio, name=name_se)(y)

        y = tf.keras.layers.Conv1D(
            filters=output_filters,
            kernel_size=1,
            strides=1,
            padding="same",
            kernel_initializer="he_normal",
            kernel_regularizer=tf.keras.regularizers.L2(1e-3),
            name=f"{name}.project" if name else None,
        )(y)

        if add_residual:
            if droprate:
                y = tf.keras.layers.Dropout(
                    droprate,
                    noise_shape=(y.shape),
                    name=f"{name}.drop" if name else None,
                )(y)
            y = tf.keras.layers.Add(name=f"{name}.res" if name else None)([x, y])
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
    skip_layers: list[tf.keras.layers.Layer | None] = []
    for i, block in enumerate(params.blocks):
        name = f"ENC{i+1}"
        for d in range(block.depth):
            y = UNext_block(
                output_filters=block.filters,
                expand_ratio=block.expand_ratio,
                kernel_size=block.kernel,
                strides=1,
                se_ratio=block.se_ratio,
                droprate=block.dropout,
                name=f"{name}.D{d+1}",
            )(y)
        # END FOR
        skip_layers.append(y if block.skip else None)

        # Downsample using strided conv
        y = tf.keras.layers.Conv1D(
            filters=block.filters,
            kernel_size=block.pool,
            strides=block.strides,
            padding="same",
            kernel_initializer="he_normal",
            kernel_regularizer=tf.keras.regularizers.L2(1e-3),
            name=f"{name}.pool",
        )(y)

        y = tf.keras.layers.LayerNormalization(
            axis=(1),
            name=f"{name}.norm",
        )(y)
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
                droprate=block.dropout,
                name=f"{name}.D{d+1}",
            )(y)
        # END FOR

        # Upsample using transposed conv
        y = tf.keras.layers.Conv1DTranspose(
            filters=block.filters,
            kernel_size=block.pool,
            strides=block.strides,
            padding="same",
            kernel_initializer="he_normal",
            kernel_regularizer=tf.keras.regularizers.L2(1e-3),
            name=f"{name}.unpool",
        )(y)

        # Skip connection
        skip_layer = skip_layers.pop()
        if skip_layer is not None:
            y = tf.keras.layers.Concatenate(name=f"{name}.S1.cat")([y, skip_layer])

            # Use conv to reduce filters
            y = tf.keras.layers.Conv1D(
                block.filters,
                kernel_size=1,  # block.kernel,
                padding="same",
                kernel_initializer="he_normal",
                kernel_regularizer=tf.keras.regularizers.L2(1e-3),
                use_bias=block.norm is None,
                name=f"{name}.S1.conv",
            )(y)

            y = tf.keras.layers.LayerNormalization(
                axis=(1),
                name=f"{name}.S1.norm",
            )(y)

            y = tf.keras.layers.Activation(
                tf.nn.relu6,
                name=f"{name}.S1.relu" if name else None,
            )(y)
        # END IF

        y = UNext_block(
            output_filters=block.filters,
            expand_ratio=block.expand_ratio,
            kernel_size=block.kernel,
            strides=1,
            se_ratio=block.se_ratio,
            droprate=block.dropout,
            name=f"{name}.D{block.depth+1}",
        )(y)

    # END FOR
    return y


def UNext(
    x: tf.Tensor,
    params: UNextParams,
    num_classes: int,
) -> tf.keras.Model:
    """Create UNext TF functional model

    Args:
        x (tf.Tensor): Input tensor
        params (UNextParams): Model parameters.
        num_classes (int, optional): # classes.

    Returns:
        tf.keras.Model: Model
    """

    y = unext_core(x, params)

    if params.include_top:
        # Add a per-point classification layer
        y = tf.keras.layers.Conv1D(
            num_classes,
            kernel_size=params.output_kernel_size,
            padding="same",
            kernel_initializer="he_normal",
            kernel_regularizer=tf.keras.regularizers.L2(1e-3),
            name="NECK.conv",
            use_bias=True,
        )(y)
        if not params.use_logits:
            y = tf.keras.layers.Softmax()(y)
        # END IF
    # END IF

    # Define the model
    model = tf.keras.Model(x, y, name=params.model_name)
    return model
