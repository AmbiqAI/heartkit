from typing import Callable

import tensorflow as tf

# import tensorflow_addons as tfa
from keras.engine.keras_tensor import KerasTensor


def make_divisible(v, divisor: int = 4, min_value=None):
    """Ensure layer has # channels divisble by divisor
       https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    Args:
        v (int): # channels
        divisor (int, optional): Divisor. Defaults to 4.
        min_value (int|None, optional): Min # channels. Defaults to None.

    Returns:
        int: # channels
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def batch_norm(name: str | None = None) -> tf.keras.layers.Layer:
    """Batch normalization layer"""
    name = name + ".bn" if name else None
    return tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name=name)


def relu(name: str | None = None) -> tf.keras.layers.Layer:
    """ReLU activation layer"""
    name = name + ".act" if name else None
    return tf.keras.layers.ReLU(name=name)


def relu6(name: str | None = None) -> tf.keras.layers.Layer:
    """Hard ReLU activation layer"""
    name = name + ".act" if name else None
    return tf.keras.layers.Activation(tf.nn.relu6, name=name)
    # return tf.keras.layers.Activation(tfa.activations.mish)
    # return tf.keras.layers.Activation(tf.nn.swish, name=name)


def gelu(name: str | None = None) -> tf.keras.layers.Layer:
    """GeLU activation layer"""
    name = name + ".act" if name else None
    return tf.keras.layers.Activation("gelu", name=name)


def hard_sigmoid(name: str | None = None) -> tf.keras.layers.Layer:
    """Hard sigmoid activation layer"""
    name = name + ".act" if name else None
    return tf.keras.layers.Activation(tf.keras.activations.hard_sigmoid, name=name)


def conv2d(
    filters: int,
    kernel_size: int | tuple[int, int] = 3,
    strides: int | tuple[int, int] = 1,
    padding: str = "same",
    use_bias: bool = False,
    name: str | None = None,
) -> tf.keras.layers.Layer:
    """2D convolutional layer

    Args:
        filters (int): # filters
        kernel_size (int | tuple[int, int], optional): Kernel size. Defaults to 3.
        strides (int | tuple[int, int], optional): Stride length. Defaults to 1.
        padding (str, optional): Padding. Defaults to "same".
        name (str|None, optional): Layer name. Defaults to None.

    Returns:
        Callable[[KerasTensor], KerasTensor]: Functional layer
    """
    name = name + ".conv" if name else None
    return tf.keras.layers.Conv2D(
        filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        use_bias=use_bias,
        kernel_initializer=tf.keras.initializers.VarianceScaling(),
        name=name,
    )


def se_block(
    ratio: int = 8, name: str | None = None
) -> Callable[[KerasTensor], KerasTensor]:
    """Squeeze & excite block

    Args:
        ratio (Expansion ratio, optional): Expansion ratio. Defaults to 8.
        name (str|None, optional): Block name. Defaults to None.

    Returns:
        Callable[[KerasTensor], KerasTensor]: Function layer
    """

    def layer(x: KerasTensor) -> KerasTensor:
        num_chan = x.shape[-1]
        # Squeeze
        name_pool = f"{name}.pool" if name else None
        name_sq = f"{name}.sq" if name else None
        y = tf.keras.layers.GlobalAveragePooling2D(name=name_pool, keepdims=True)(x)
        y = conv2d(num_chan // ratio, kernel_size=(1, 1), use_bias=True, name=name_sq)(
            y
        )
        y = relu6(name=name_sq)(y)
        # Excite
        name_ex = f"{name}.ex" if name else None
        y = conv2d(num_chan, kernel_size=(1, 1), use_bias=True, name=name_ex)(y)
        y = hard_sigmoid(name=name_ex)(y)
        y = tf.keras.layers.Multiply()([x, y])
        return y

    return layer


def mbconv_block(
    output_filters: int,
    expand_ratio: float = 1,
    kernel_size: int | tuple[int, int] = 3,
    strides: int | tuple[int, int] = 1,
    se_ratio: float = 8,
    droprate: float = 0,
    name: str | None = None,
) -> Callable[[KerasTensor], KerasTensor]:
    """MBConv block w/ expansion and SE

    Args:
        output_filters (int): # output filter channels
        expand_ratio (float, optional): Expansion ratio. Defaults to 1.
        kernel_size (int | tuple[int, int], optional): Kernel size. Defaults to 3.
        strides (int | tuple[int, int], optional): Stride length. Defaults to 1.
        se_ratio (float, optional): SE ratio. Defaults to 8.
        droprate (float, optional): Drop rate. Defaults to 0.
        name (str|None, optional): Block name. Defaults to None.

    Returns:
        Callable[[KerasTensor], KerasTensor]: Functional layer
    """

    def layer(x: KerasTensor) -> KerasTensor:
        input_filters = x.shape[-1]
        add_residual = input_filters == output_filters and (
            strides == 1 if isinstance(strides, int) else strides[0] == 1
        )
        # Expand: narrow -> wide
        if expand_ratio > 1:
            name_ex = f"{name}.exp" if name else None
            filters = int(input_filters * expand_ratio)
            y = conv2d(filters, kernel_size=(1, 1), strides=(1, 1), name=name_ex)(x)
            y = batch_norm(name=name_ex)(y)
            y = relu6(name=name_ex)(y)
        else:
            y = x

        # Apply: wide -> wide
        name_dp = f"{name}.dp" if name else None
        y = tf.keras.layers.DepthwiseConv2D(
            kernel_size=kernel_size,
            strides=strides,
            padding="same",
            use_bias=False,
            name=name_dp,
        )(y)
        y = batch_norm(name=name_dp)(y)
        y = relu6(name=name_dp)(y)

        # SE: wide -> wide
        if se_ratio:
            name_se = f"{name}.se" if name else None
            y = se_block(ratio=se_ratio * expand_ratio, name=name_se)(y)

        # Reduce: wide -> narrow
        name_red = f"{name}.red" if name else None
        y = conv2d(
            output_filters,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding="same",
            name=name_red,
        )(y)
        y = batch_norm(name=name_red)(y)
        # No activation

        if add_residual:
            name_res = f"{name}.res" if name else None
            if droprate > 0:
                y = tf.keras.layers.Dropout(droprate, noise_shape=(None, 1, 1, 1))(y)
            y = tf.keras.layers.add([x, y], name=name_res)
        return y

    return layer
