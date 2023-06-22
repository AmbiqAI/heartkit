

import tensorflow as tf
from keras.engine.keras_tensor import KerasTensor
from pydantic import BaseModel, Field

from .blocks import batch_norm, multiresnet_block, glu

class MultiresNetParams(BaseModel):
    """EfficientNet parameters"""
    d_model: int = Field(256, description="Model depth")
    n_layers: int = Field(4, description="Number of layers")
    dropout: float = Field(default=0.2, description="Dropout rate")
    kernel_size: int = Field(default=2, description="Kernel size")
    depth: int|None = Field(default=None, description="Depth")
    seq_len: int|None = Field(default=None, description="Sequence length")
    activation_scaling: float = Field(default=2.0, description="Activation scaling")
    include_top: bool = Field(default=True, description="Include top")
    model_name: str = Field(default="MultiresNet", description="Model name")


def MultiresNet(
    x: KerasTensor,
    params: MultiresNetParams,
    num_classes: int | None = None,
):

    # Apply stem
    y = tf.keras.layers.Conv1D(params.d_model, kernel_size=1)(x)

    # Apply multiresnet blocks
    for _ in range(params.n_layers):
        y_res = y
        y = multiresnet_block(
            d_model=params.d_model,
            kernel_size=params.kernel_size,
            depth=params.depth,
            seq_len=params.seq_len,
            droprate=params.dropout,

        )(y)
        # Mix channels
        y = tf.keras.layers.Conv1D(
            int(params.activation_scaling * params.d_model), 1
        )(y)
        y = glu()(y)
        y = tf.keras.layers.Dropout(params.dropout)(y)
        y = tf.keras.layers.Add()([y, y_res])
        y = batch_norm()(y)
    # END FOR

    if params.include_top:
        name = "top"
        y = tf.keras.layers.GlobalAveragePooling1D(name=f"{name}.pool")(y)
        if 0 < params.dropout < 1:
            y = tf.keras.layers.Dropout(params.dropout)(y)
        y = tf.keras.layers.Dense(num_classes, name=name)(y)
    model = tf.keras.Model(x, y, name=params.model_name)
    return model
