import math

import tensorflow as tf
from pydantic import BaseModel, Field

from .blocks import SharedWeightsConv, batch_norm, glu, relu6
from .defines import KerasLayer


class MultiresNetParams(BaseModel):
    """Multiresnet parameters"""

    d_model: int = Field(256, description="Model depth")
    n_layers: int = Field(4, description="Number of layers")
    dropout: float = Field(default=0.2, description="Dropout rate")
    kernel_size: int = Field(default=2, description="Kernel size")
    depth: int | None = Field(default=None, description="Depth")
    seq_len: int | None = Field(default=None, description="Sequence length")
    activation_scaling: float = Field(default=2.0, description="Activation scaling")
    include_top: bool = Field(default=True, description="Include top")
    model_name: str = Field(default="MultiresNet", description="Model name")


def multiresnet_block(
    d_model: int,
    kernel_size: int | tuple[int, int] = 3,
    depth: int | None = None,
    wavelet_init=None,
    tree_select: str = "fading",
    seq_len: int | None = None,
    droprate: float = 0,
    memory_size: int | None = None,
    indep_res_init: bool = False,
    name: str | None = None,
) -> KerasLayer:
    """Multiresnet block

    Args:
        name (str|None, optional): Block name. Defaults to None.

    Returns:
       KerasLayer: Functional layer

    """

    if depth is None and seq_len is None:
        raise ValueError("Either depth or seq_len must be specified")

    if depth is None:
        depth = math.ceil(math.log2((seq_len - 1) / (kernel_size - 1) + 1))

    # if tree_select == 'fading':
    #     m = depth + 1
    # elif memory_size is not None:
    #     m = memory_size
    # else:
    #     raise ValueError("Either tree_select is fading or memory_size must be specified")

    if wavelet_init is not None:
        import pywt  # pylint: disable=import-outside-toplevel,import-error  # type: ignore

        wavelet = pywt.Wavelet(wavelet_init)
        h0 = tf.convert_to_tensor(wavelet.dec_lo[::-1])
        h1 = tf.convert_to_tensor(wavelet.dec_hi[::-1])
        h0 = tf.tile(tf.reshape(h0, (1, 1, -1)), [d_model, 1, 1])
        h1 = tf.tile(tf.reshape(h1, (1, 1, -1)), [d_model, 1, 1])
    elif kernel_size is not None:
        h0 = "glorot_uniform"
        h1 = "glorot_uniform"
    else:
        raise ValueError("Either wavelet_init or kernel_size must be specified")

    def layer(x: tf.Tensor) -> tf.Tensor:
        if tree_select == "fading":
            res_lo = x
            y = tf.keras.layers.DepthwiseConv1D(
                kernel_size=1,
            )(x)
            conv_hi = tf.keras.layers.Conv1D(
                filters=d_model,
                kernel_size=kernel_size,
                kernel_initializer=h1,
                dilation_rate=1,
                padding="valid",
                groups=x.shape[2],
            )
            conv_lo = tf.keras.layers.Conv1D(
                filters=d_model,
                kernel_size=kernel_size,
                kernel_initializer=h0,
                dilation_rate=1,
                padding="valid",
                groups=x.shape[2],
            )
            dilation = 1
            for _ in range(depth, 0, -1):
                padding = dilation * (kernel_size - 1)
                res_lo_pad = tf.keras.layers.ZeroPadding1D((padding, 0))(res_lo)
                res_hi = SharedWeightsConv(conv_hi, dilation_rate=dilation)(res_lo_pad)
                res_lo = SharedWeightsConv(conv_lo, dilation_rate=dilation)(res_lo_pad)
                res_hi = tf.keras.layers.DepthwiseConv1D(kernel_size=1)(res_hi)
                y = tf.keras.layers.add([y, res_hi])
                dilation *= 2
            # END FOR
            res_lo = tf.keras.layers.DepthwiseConv1D(
                kernel_size=1,
            )(res_lo)
            y = tf.keras.layers.add([y, res_lo])

        elif tree_select == "uniform":
            raise NotImplementedError("tree_select == 'uniform' is not implemented yet")

        y = tf.keras.layers.Dropout(droprate)(y)
        y = relu6()(y)
        return y

    return layer


def MultiresNet(
    x: tf.Tensor,
    params: MultiresNetParams,
    num_classes: int | None = None,
):
    """MultiresNet architecture"""
    y = x

    # Apply stem
    y = tf.keras.layers.Conv1D(params.d_model, kernel_size=1)(y)

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
        y = tf.keras.layers.Conv1D(int(params.activation_scaling * params.d_model), 1)(y)
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
