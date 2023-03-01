""" EfficientNet """

import tensorflow as tf
from keras.engine.keras_tensor import KerasTensor
from pydantic import BaseModel, Field

from .blocks import batch_norm, conv2d, make_divisible, mbconv_block, relu6


class MBConvParam(BaseModel):
    """MBConv parameters"""

    filters: int = Field(..., description="# filters")
    depth: int = Field(default=1, description="Layer depth")
    ex_ratio: float = Field(default=1, description="Expansion ratio")
    kernel_size: int | tuple[int, int] = Field(default=3, description="Kernel size")
    strides: int | tuple[int, int] = Field(default=1, description="Stride size")
    se_ratio: float = Field(default=8, description="Squeeze Excite ratio")
    droprate: float = Field(default=0, description="Drop rate")


def EfficientNetV2(
    x: KerasTensor,
    blocks: list[MBConvParam],
    input_filters: int = 0,
    input_strides: int | tuple[int, int] = 1,
    input_kernel_size: int | tuple[int, int] = 3,
    output_filters: int = 0,
    include_top: bool = True,
    dropout: float = 0.2,
    num_classes: int = 2,
    drop_connect_rate: float = 0,
    model_name: str = "EfficientNetV2",
) -> tf.keras.Model:
    """Create EfficientNet V2 TF functional model

    Args:
        x (KerasTensor): Input tensor
        blocks (list[MBConvParam]): List of MBConv parameters
        input_filters (int, optional): Initial # input filter channels. Defaults to 0.
        input_strides (int | tuple[int, int], optional): # input stride length. Defaults to 1.
        input_kernel_size (int | tuple[int, int], optional): Input kernel size. Defaults to 3.
        output_filters (int, optional): Output filter channels. Defaults to 0.
        include_top (bool, optional): Include top layer. Defaults to True.
        dropout (float, optional): Dropout amount. Defaults to 0.2.
        num_classes (int, optional): # classes. Defaults to 2.
        drop_connect_rate (float, optional): Drop connect rate. Defaults to 0.
        model_name (str, optional): Model name. Defaults to "EfficientNetV2".

    Returns:
        tf.keras.Model: Model
    """
    if input_filters > 0:
        name = "stem"
        filters = make_divisible(input_filters, 8)
        y = conv2d(filters, kernel_size=input_kernel_size, strides=input_strides, name=name)(x)
        y = batch_norm(name=name)(y)
        y = relu6(name=name)(y)
    else:
        y = x

    global_block_id = 0
    total_blocks = sum((b.depth for b in blocks))
    for i, block in enumerate(blocks):
        filters = make_divisible(block.filters, 8)
        for d in range(block.depth):
            name = f"stage{i+1}.mbconv{d+1}"
            block_drop_rate = drop_connect_rate * global_block_id / total_blocks
            y = mbconv_block(
                filters,
                block.ex_ratio,
                block.kernel_size,
                block.strides if d == 0 else (1, 1),
                block.se_ratio,
                droprate=block_drop_rate,
                name=name,
            )(y)
            global_block_id += 1
        # END FOR
    # END FOR
    if output_filters:
        name = "neck"
        filters = make_divisible(output_filters, 8)
        y = conv2d(filters, kernel_size=(1, 1), strides=(1, 1), padding="same", name=name)(y)
        y = batch_norm(name=name)(y)
        y = relu6(name=name)(y)

    if include_top:
        name = "top"
        y = tf.keras.layers.GlobalAveragePooling2D(name=f"{name}.pool")(y)
        if 0 < dropout < 1:
            y = tf.keras.layers.Dropout(dropout)(y)
        y = tf.keras.layers.Dense(num_classes, name=name)(y)
    model = tf.keras.Model(x, y, name=model_name)
    return model
