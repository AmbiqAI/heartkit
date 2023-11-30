""" EfficientNet https://arxiv.org/abs/2104.00298"""
import tensorflow as tf
from pydantic import BaseModel, Field

from .blocks import batch_norm, conv2d, mbconv_block, relu6
from .defines import KerasLayer, MBConvParams
from .utils import make_divisible


class EfficientNetParams(BaseModel):
    """EfficientNet parameters"""

    blocks: list[MBConvParams] = Field(default_factory=list, description="EfficientNet blocks")
    input_filters: int = Field(default=0, description="Input filters")
    input_kernel_size: int | tuple[int, int] = Field(default=3, description="Input kernel size")
    input_strides: int | tuple[int, int] = Field(default=2, description="Input stride")
    output_filters: int = Field(default=0, description="Output filters")
    include_top: bool = Field(default=True, description="Include top")
    dropout: float = Field(default=0.2, description="Dropout rate")
    drop_connect_rate: float = Field(default=0.2, description="Drop connect rate")
    model_name: str = Field(default="EfficientNetV2", description="Model name")


def efficientnet_core(blocks: list[MBConvParams], drop_connect_rate: float = 0) -> KerasLayer:
    """EfficientNet core

    Args:
        blocks (list[MBConvParam]): MBConv params
        drop_connect_rate (float, optional): Drop connect rate. Defaults to 0.

    Returns:
        KerasLayer: Core
    """

    def layer(x: tf.Tensor) -> tf.Tensor:
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

    # END DEF
    return layer


def EfficientNetV2(
    x: tf.Tensor,
    params: EfficientNetParams,
    num_classes: int | None = None,
) -> tf.keras.Model:
    """Create EfficientNet V2 TF functional model

    Args:
        x (tf.Tensor): Input tensor
        params (EfficientNetParams): Model parameters.
        num_classes (int, optional): # classes.

    Returns:
        tf.keras.Model: Model
    """

    # Force input to be 4D (add dummy dimension)
    requires_reshape = len(x.shape) == 3
    if requires_reshape:
        y = tf.keras.layers.Reshape((1,) + x.shape[1:])(x)
    else:
        y = x
    # END IF

    # Stem
    if params.input_filters > 0:
        name = "stem"
        filters = make_divisible(params.input_filters, 8)
        y = conv2d(
            filters,
            kernel_size=params.input_kernel_size,
            strides=params.input_strides,
            name=name,
        )(y)
        y = batch_norm(name=name)(y)
        y = relu6(name=name)(y)
    # END IF

    y = efficientnet_core(blocks=params.blocks, drop_connect_rate=params.drop_connect_rate)(y)

    if params.output_filters:
        name = "neck"
        filters = make_divisible(params.output_filters, 8)
        y = conv2d(filters, kernel_size=(1, 1), strides=(1, 1), padding="same", name=name)(y)
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
