"""TCN """
from typing import Literal

import tensorflow as tf
from pydantic import BaseModel, Field

from .blocks import se_block
from .defines import KerasLayer


class TcnBlockParams(BaseModel):
    """TCN block parameters"""

    depth: int = Field(default=1, description="Layer depth")
    branch: int = Field(default=1, description="Number of branches")
    filters: int = Field(..., description="# filters")
    kernel: int | tuple[int, int] = Field(default=3, description="Kernel size")
    dilation: int | tuple[int, int] = Field(default=1, description="Dilation rate")
    ex_ratio = Field(default=1, description="Expansion ratio")
    se_ratio: float = Field(default=0, description="Squeeze and excite ratio")
    dropout: float | None = Field(default=None, description="Dropout rate")
    norm: Literal["batch", "layer"] | None = Field(default="layer", description="Normalization type")


class TcnParams(BaseModel):
    """TCN parameters"""

    input_kernel: int | tuple[int, int] | None = Field(default=None, description="Input kernel size")
    input_norm: Literal["batch", "layer"] | None = Field(default="layer", description="Input normalization type")
    block_type: Literal["lg", "mb", "sm"] = Field(default="mb", description="Block type")
    blocks: list[TcnBlockParams] = Field(default_factory=list, description="UNext blocks")
    output_kernel: int | tuple[int, int] = Field(default=3, description="Output kernel size")
    include_top: bool = Field(default=True, description="Include top")
    use_logits: bool = Field(default=True, description="Use logits")
    model_name: str = Field(default="UNext", description="Model name")


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
            return tf.keras.layers.BatchNormalization(axis=-1, name=f"{name}.BN")(x)
        if norm == "layer":
            return tf.keras.layers.LayerNormalization(axis=(1, 2), name=f"{name}.LN")(x)
        return x

    return layer


def tcn_block_lg(params: TcnBlockParams, name: str) -> KerasLayer:
    """TCN large block

    Args:
        params (TcnBlockParams): Parameters
        name (str): Name

    Returns:
        KerasLayer: Layer
    """

    def layer(x: tf.Tensor) -> tf.Tensor:
        """TCN block layer"""
        y = x

        for d in range(params.depth):
            lcl_name = f"{name}.D{d+1}"
            y_skip = y

            y = tf.keras.layers.Conv2D(
                filters=params.filters,
                kernel_size=params.kernel,
                strides=(1, 1),
                padding="same",
                use_bias=False,
                dilation_rate=params.dilation,
                kernel_initializer="he_normal",
                kernel_regularizer=tf.keras.regularizers.L2(1e-3),
                name=f"{lcl_name}.CN1",
            )(y)
            y = norm_layer(params.norm, f"{lcl_name}.CN1")(y)

            y = tf.keras.layers.Conv2D(
                filters=params.filters,
                kernel_size=params.kernel,
                strides=(1, 1),
                padding="same",
                use_bias=params.norm is None,
                dilation_rate=params.dilation,
                kernel_initializer="he_normal",
                kernel_regularizer=tf.keras.regularizers.L2(1e-3),
                name=f"{lcl_name}.CN2",
            )(y)
            y = norm_layer(params.norm, f"{lcl_name}.CN2")(y)

            if y_skip.shape[-1] == y.shape[-1]:
                y = tf.keras.layers.Add(name=f"{lcl_name}.ADD")([y, y_skip])

            y = tf.keras.layers.Activation("relu6", name=f"{lcl_name}.RELU")(y)

            # Squeeze and excite
            if params.se_ratio > 0:
                y = se_block(ratio=params.se_ratio, name=f"{lcl_name}.SE")(y)
            # END IF

            if params.dropout and params.dropout > 0:
                y = tf.keras.layers.SpatialDropout2D(rate=params.dropout, name=f"{lcl_name}.DROP")(y)
            # END IF

        # END FOR
        return y

    return layer


def tcn_block_mb(params: TcnBlockParams, name: str) -> KerasLayer:
    """TCN mbconv block

    Args:
        params (TcnBlockParams): Parameters
        name (str): Name

    Returns:
        KerasLayer: Layer
    """

    def layer(x: tf.Tensor) -> tf.Tensor:
        """TCN block layer"""
        y = x
        y_skip = y
        for d in range(params.depth):
            lcl_name = f"{name}.D{d+1}"

            if params.ex_ratio != 1:
                y = tf.keras.layers.Conv2D(
                    filters=int(params.filters * params.ex_ratio),
                    kernel_size=(1, 1),
                    strides=(1, 1),
                    padding="same",
                    use_bias=params.norm is None,
                    kernel_initializer="he_normal",
                    kernel_regularizer=tf.keras.regularizers.L2(1e-3),
                    name=f"{lcl_name}.EX.CN",
                )(y)
                y = norm_layer(params.norm, f"{lcl_name}.EX")(y)
                y = tf.keras.layers.Activation("relu6", name=f"{lcl_name}.EX.RELU")(y)
            # END IF

            branches = []
            for b in range(params.branch):
                yb = y
                yb = tf.keras.layers.DepthwiseConv2D(
                    kernel_size=params.kernel,
                    strides=(1, 1),
                    padding="same",
                    use_bias=params.norm is None,
                    dilation_rate=params.dilation,
                    depthwise_initializer="he_normal",
                    depthwise_regularizer=tf.keras.regularizers.L2(1e-3),
                    name=f"{lcl_name}.DW.B{b+1}.CN",
                )(yb)
                yb = norm_layer(params.norm, f"{lcl_name}.DW.B{b+1}")(yb)
                branches.append(yb)
            # END FOR

            if params.branch > 1:
                y = tf.keras.layers.Add(name=f"{lcl_name}.DW.ADD")(branches)
            else:
                y = branches[0]
            # END IF

            y = tf.keras.layers.Activation("relu6", name=f"{lcl_name}.DW.RELU")(y)

            # Squeeze and excite
            if params.se_ratio and y.shape[-1] // params.se_ratio > 0:
                y = se_block(ratio=params.se_ratio, name=f"{lcl_name}.SE")(y)
            # END IF

            branches = []
            for b in range(params.branch):
                yb = y
                yb = tf.keras.layers.Conv2D(
                    filters=params.filters,
                    kernel_size=(1, 1),
                    strides=(1, 1),
                    padding="same",
                    use_bias=params.norm is None,
                    kernel_initializer="he_normal",
                    kernel_regularizer=tf.keras.regularizers.L2(1e-3),
                    name=f"{lcl_name}.PW.B{b+1}.CN",
                )(yb)
                yb = norm_layer(params.norm, f"{lcl_name}.PW.B{b+1}")(yb)
                branches.append(yb)
            # END FOR

            if params.branch > 1:
                y = tf.keras.layers.Add(name=f"{lcl_name}.PW.ADD")(branches)
            else:
                y = branches[0]
            # END IF

            y = tf.keras.layers.Activation("relu6", name=f"{lcl_name}.PW.RELU")(y)
        # END FOR

        # Skip connection
        if y_skip.shape[-1] == y.shape[-1]:
            y = tf.keras.layers.Add(name=f"{name}.ADD")([y, y_skip])

        if params.dropout and params.dropout > 0:
            y = tf.keras.layers.SpatialDropout2D(rate=params.dropout, name=f"{name}.DROP")(y)
        # END IF
        return y

    return layer


def tcn_block_sm(params: TcnBlockParams, name: str) -> KerasLayer:
    """TCN small block

    Args:
        params (TcnBlockParams): Parameters
        name (str): Name

    Returns:
        KerasLayer: Layer
    """

    def layer(x: tf.Tensor) -> tf.Tensor:
        """TCN block layer"""
        y = x
        y_skip = y
        for d in range(params.depth):
            lcl_name = f"{name}.D{d+1}"
            branches = []
            for b in range(params.branch):
                yb = y
                yb = tf.keras.layers.DepthwiseConv2D(
                    kernel_size=params.kernel,
                    strides=(1, 1),
                    padding="same",
                    use_bias=params.norm is None,
                    dilation_rate=params.dilation,
                    depthwise_initializer="he_normal",
                    depthwise_regularizer=tf.keras.regularizers.L2(1e-3),
                    name=f"{lcl_name}.DW.B{b+1}.CN",
                )(yb)
                yb = norm_layer(params.norm, f"{lcl_name}.DW.B{b+1}")(yb)
                branches.append(yb)
            # END FOR

            if params.branch > 1:
                y = tf.keras.layers.Add(name=f"{lcl_name}.DW.ADD")(branches)
            else:
                y = branches[0]
            # END IF

            y = tf.keras.layers.Activation("relu6", name=f"{lcl_name}.DW.RELU")(y)

            branches = []
            for b in range(params.branch):
                yb = y
                yb = tf.keras.layers.Conv2D(
                    filters=params.filters,
                    kernel_size=(1, 1),
                    strides=(1, 1),
                    padding="same",
                    # groups=int(params.se_ratio) if params.se_ratio > 0 else 1,
                    use_bias=params.norm is None,
                    kernel_initializer="he_normal",
                    kernel_regularizer=tf.keras.regularizers.L2(1e-3),
                    name=f"{lcl_name}.PW.B{b+1}.CN",
                )(yb)
                yb = norm_layer(params.norm, f"{lcl_name}.PW.B{b+1}")(yb)
                branches.append(yb)
            # END FOR

            if params.branch > 1:
                y = tf.keras.layers.Add(name=f"{lcl_name}.PW.ADD")(branches)
            else:
                y = branches[0]
            # END IF

            y = tf.keras.layers.Activation("relu6", name=f"{lcl_name}.PW.RELU")(y)
        # END FOR

        # Squeeze and excite
        if y.shape[-1] // params.se_ratio > 1:
            y = se_block(ratio=params.se_ratio, name=f"{name}.SE")(y)
        # END IF

        # Skip connection
        if y_skip.shape[-1] == y.shape[-1]:
            y = tf.keras.layers.Add(name=f"{name}.ADD")([y, y_skip])

        if params.dropout and params.dropout > 0:
            y = tf.keras.layers.SpatialDropout2D(rate=params.dropout, name=f"{name}.DROP")(y)
        # END IF
        return y

    return layer


def tcn_core(params: TcnParams) -> KerasLayer:
    """TCN core

    Args:
        params (TcnParams): Parameters

    Returns:
        KerasLayer: Layer
    """
    if params.block_type == "lg":
        tcn_block = tcn_block_lg
    elif params.block_type == "mb":
        tcn_block = tcn_block_mb
    elif params.block_type == "sm":
        tcn_block = tcn_block_sm
    else:
        raise ValueError(f"Invalid block type: {params.block_type}")

    def layer(x: tf.Tensor) -> tf.Tensor:
        y = x
        for i, block in enumerate(params.blocks):
            name = f"B{i+1}"
            y = tcn_block(params=block, name=name)(y)
        # END IF
        return y

    return layer


def Tcn(
    x: tf.Tensor,
    params: TcnParams,
    num_classes: int,
) -> tf.keras.Model:
    """TCN model

    Args:
        x (tf.Tensor): Input tensor
        params (TcnParams): Parameters
        num_classes (int): Number of classes

    Returns:
        tf.keras.Model: Model
    """
    requires_reshape = len(x.shape) == 3
    if requires_reshape:
        y = tf.keras.layers.Reshape((1,) + x.shape[1:])(x)
    else:
        y = x

    # Encode each channel separately
    if params.input_kernel:
        y = tf.keras.layers.DepthwiseConv2D(
            kernel_size=params.input_kernel, use_bias=params.input_norm is None, name="ENC.CN", padding="same"
        )(y)
        y = norm_layer(params.input_norm, "ENC")(y)
    # END IF

    y = tcn_core(params)(y)

    if params.include_top:
        # Add a per-point classification layer
        y = tf.keras.layers.Conv2D(
            num_classes,
            kernel_size=params.output_kernel,
            padding="same",
            name="NECK.conv",
            use_bias=True,
        )(y)
        if not params.use_logits:
            y = tf.keras.layers.Softmax()(y)
        # END IF
    # END IF

    if requires_reshape:
        y = tf.keras.layers.Reshape(y.shape[2:])(y)

    # Define the model
    model = tf.keras.Model(x, y, name=params.model_name)
    return model
