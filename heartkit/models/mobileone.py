"""MobileOne https://arxiv.org/abs/2206.04040"""

import keras
import tensorflow as tf
from pydantic import BaseModel, Field

from .blocks import batch_norm, conv2d, relu6, se_block
from .defines import KerasLayer


class MobileOneBlockParams(BaseModel):
    """MobileOne block parameters"""

    filters: int = Field(..., description="# filters")
    depth: int = Field(default=1, description="Layer depth")
    kernel_size: int | tuple[int, int] = Field(default=3, description="Kernel size")
    strides: int | tuple[int, int] = Field(default=1, description="Stride size")
    padding: int | tuple[int, int] = Field(default=0, description="Padding size")
    se_ratio: float = Field(default=8, description="Squeeze Excite ratio")
    se_depth: int = Field(default=0, description="Depth length to apply SE")
    num_conv_branches: int = Field(default=2, description="# conv branches")


class MobileOneParams(BaseModel):
    """MobileOne parameters"""

    blocks: list[MobileOneBlockParams] = Field(default_factory=list, description="MobileOne blocks")

    input_filters: int = Field(default=3, description="Input filters")
    input_kernel_size: int | tuple[int, int] = Field(default=3, description="Input kernel size")
    input_strides: int | tuple[int, int] = Field(default=2, description="Input stride")
    input_padding: int | tuple[int, int] = Field(default=1, description="Input padding")

    # output_filters: int = Field(default=0, description="Output filters")
    include_top: bool = Field(default=True, description="Include top")
    dropout: float = Field(default=0.2, description="Dropout rate")
    # drop_connect_rate: float = Field(default=0.2, description="Drop connect rate")
    model_name: str = Field(default="MobileOne", description="Model name")


def mobileone_block(
    output_filters: int,
    kernel_size: int | tuple[int, int] = 3,
    strides: int | tuple[int, int] = 1,
    padding: int | tuple[int, int] = 0,
    groups: int = 1,
    dilation: int = 1,
    inference_mode: bool = False,
    se_ratio: int = 0,
    num_conv_branches: int = 1,
    name: str | None = None,
) -> KerasLayer:
    """MBConv block w/ expansion and SE

    Args:
        output_filters (int): # output filter channels
        kernel_size (int | tuple[int, int], optional): Kernel size. Defaults to 3.
        strides (int | tuple[int, int], optional): Stride length. Defaults to 1.
        se_ratio (float, optional): SE ratio. Defaults to 8.
        name (str|None, optional): Block name. Defaults to None.

    Returns:
        KerasLayer: Functional layer
    """

    def layer(x: tf.Tensor) -> tf.Tensor:
        input_filters = x.shape[-1]
        stride_len = strides if isinstance(strides, int) else sum(strides) / len(strides)
        kernel_len = kernel_size if isinstance(kernel_size, int) else sum(kernel_size) / len(kernel_size)
        is_downsample = stride_len > 1
        is_depthwise = groups > 1 and groups == input_filters
        has_skip_branch = output_filters == input_filters and stride_len == 1

        if inference_mode:
            y = keras.layers.ZeroPadding2D(padding=padding)(x)
            y = conv2d(
                output_filters,
                kernel_size=kernel_size,
                strides=strides,
                padding="valid",
                dilation=dilation,
                groups=groups,
                use_bias=True,
                name=name,
            )(y)
            if se_ratio > 0:
                name_se = f"{name}.se" if name else None
                y = se_block(ratio=se_ratio, name=name_se)(y)
            # END IF
            y = relu6(name=name)(y)
            return y
        # END IF

        branches = []

        # Skip branch
        if has_skip_branch:
            name_skip = f"{name}.skip" if name else None
            y_skip = batch_norm(name=name_skip)(x)
            branches.append(y_skip)
        # END IF

        # Either groups is input_filters or is 1

        # Scale branch
        if kernel_len > 1:
            name_scale = f"{name}.scale" if name else None
            if is_depthwise:
                y_scale = keras.layers.DepthwiseConv2D(
                    kernel_size=(1, 1),
                    strides=(1, 1),  # strides,
                    padding="valid",
                    use_bias=False,
                    depthwise_initializer="he_normal",
                    name=f"{name_scale}.conv" if name_scale else None,
                )(x)
                y_scale = batch_norm(name=name_scale)(y_scale)
                if is_downsample:
                    y_scale = keras.layers.MaxPool2D(pool_size=strides, padding="same")(y_scale)
            else:
                y_scale = keras.layers.Conv2D(
                    output_filters,
                    kernel_size=(1, 1),
                    strides=strides,
                    padding="valid",
                    groups=groups,
                    use_bias=False,
                    kernel_initializer="he_normal",
                    name=f"{name_scale}.conv" if name_scale else None,
                )(x)
                y_scale = batch_norm(name=name_scale)(y_scale)
            branches.append(y_scale)
        # END IF

        # Other branches
        yp = keras.layers.ZeroPadding2D(padding=padding)(x)
        for b in range(num_conv_branches):
            name_branch = f"{name}.branch{b+1}" if name else None
            if is_depthwise:
                y_branch = keras.layers.DepthwiseConv2D(
                    kernel_size=kernel_size,
                    strides=(1, 1),
                    padding="valid",
                    use_bias=False,
                    depthwise_initializer="he_normal",
                    name=f"{name_branch}.conv" if name else None,
                )(yp)
                y_branch = batch_norm(name=name_branch)(y_branch)
                if is_downsample:
                    y_branch = keras.layers.MaxPool2D(pool_size=strides, padding="same")(y_branch)
            else:
                y_branch = keras.layers.Conv2D(
                    output_filters,
                    kernel_size=kernel_size,
                    strides=strides,
                    groups=groups,
                    padding="valid",
                    use_bias=False,
                    kernel_initializer="he_normal",
                    name=f"{name_branch}.conv" if name else None,
                )(yp)
                y_branch = batch_norm(name=name_branch)(y_branch)
            branches.append(y_branch)
        # END FOR

        # Merge branches
        y = keras.layers.Add(name=f"{name}.add" if name else None)(branches)

        # Squeeze-Excite block
        if se_ratio > 0:
            name_se = f"{name}.se" if name else None
            y = se_block(ratio=se_ratio, name=name_se)(y)
        # END IF
        y = relu6(name=name)(y)
        return y

    # END DEF
    return layer


def MobileOne(
    x: tf.Tensor,
    params: MobileOneParams,
    num_classes: int | None = None,
    inference_mode: bool = False,
) -> keras.Model:
    """Create MobileOne TF functional model

    Args:
        x (tf.Tensor): Input tensor
        params (MobileOneParams): Model parameters.
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

    y = mobileone_block(
        output_filters=params.input_filters,
        kernel_size=params.input_kernel_size,
        strides=params.input_strides,
        padding=params.input_padding,
        groups=1,
        inference_mode=inference_mode,
        name=f"M0.B{0}.D{0}.DW",
    )(y)

    for b, block in enumerate(params.blocks):
        for d in range(block.depth):
            se_ratio = block.se_ratio if d >= block.depth - block.se_depth else 0
            # Depthwise block
            y = mobileone_block(
                output_filters=y.shape[-1],
                kernel_size=block.kernel_size,
                strides=block.strides if d == 0 else (1, 1),
                padding=block.padding,
                groups=y.shape[-1],
                inference_mode=inference_mode,
                se_ratio=se_ratio,
                num_conv_branches=block.num_conv_branches,
                name=f"M1.B{b+1}.D{d+1}.DW",
            )(y)

            # Pointwise block
            y = mobileone_block(
                output_filters=block.filters,
                kernel_size=(1, 1),
                strides=(1, 1),
                padding=(0, 0),
                groups=1,
                inference_mode=inference_mode,
                se_ratio=se_ratio,
                num_conv_branches=block.num_conv_branches,
                name=f"M1.B{b+1}.D{d+1}.PW",
            )(y)
        # END FOR
    # END FOR

    if params.include_top:
        name = "top"
        y = keras.layers.GlobalAveragePooling2D(name=f"{name}.pool")(y)
        if 0 < params.dropout < 1:
            y = keras.layers.Dropout(params.dropout)(y)
        y = keras.layers.Dense(num_classes, name=name)(y)

    model = keras.Model(x, y, name=params.model_name)

    return model


def MobileOneU0(x, num_classes):
    """micro-0 MobileOne network"""
    return MobileOne(
        x=x,
        params=MobileOneParams(
            input_filters=16,
            input_kernel_size=(1, 7),
            input_strides=(1, 2),
            input_padding=(0, 3),
            blocks=[
                MobileOneBlockParams(
                    filters=32,
                    depth=2,
                    kernel_size=(1, 5),
                    strides=(1, 2),
                    padding=(0, 2),
                    se_ratio=0,
                    se_depth=0,
                    num_conv_branches=3,
                ),
                MobileOneBlockParams(
                    filters=64,
                    depth=3,
                    kernel_size=(1, 3),
                    strides=(1, 2),
                    padding=(0, 1),
                    se_ratio=2,
                    se_depth=1,
                    num_conv_branches=3,
                ),
                MobileOneBlockParams(
                    filters=128,
                    depth=3,
                    kernel_size=(1, 3),
                    strides=(1, 2),
                    padding=(0, 1),
                    se_ratio=4,
                    se_depth=1,
                    num_conv_branches=3,
                ),
                MobileOneBlockParams(
                    filters=256,
                    depth=2,
                    kernel_size=(1, 3),
                    strides=(1, 2),
                    padding=(0, 1),
                    se_ratio=4,
                    se_depth=1,
                    num_conv_branches=3,
                ),
            ],
        ),
        num_classes=num_classes,
        inference_mode=False,
    )
