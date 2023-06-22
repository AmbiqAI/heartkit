import math
from typing import Callable

import tensorflow as tf
import tensorflow_addons as tfa
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


def glu(dim: int = -1) -> tf.keras.layers.Layer:
    def layer(x: KerasTensor) -> KerasTensor:
        out, gate = tf.split(x, num_or_size_splits=2, axis=dim)
        gate = tf.sigmoid(gate)
        x = tf.multiply(out, gate)
        return x
    return layer

def relu(name: str | None = None) -> tf.keras.layers.Layer:
    """ReLU activation layer"""
    name = name + ".act" if name else None
    return tf.keras.layers.ReLU(name=name)


def relu6(name: str | None = None) -> tf.keras.layers.Layer:
    """Hard ReLU activation layer"""
    name = name + ".act" if name else None
    return tf.keras.layers.Activation(tf.nn.relu6, name=name)


def mish(name: str | None = None) -> tf.keras.layers.Layer:
    """Mish activation layer"""
    return tf.keras.layers.Activation(tfa.activations.mish)


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
        use_bias (bool, optional): Add bias. Defaults to False.
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


def conv1d(
    filters: int,
    kernel_size: int = 3,
    strides: int = 1,
    padding: str = "same",
    use_bias: bool = False,
    name: str | None = None,
) -> tf.keras.layers.Layer:
    """1D convolutional layer using 2D convolutional layer

    Args:
        filters (int): # filters
        kernel_size (int, optional): Kernel size. Defaults to 3.
        strides (int, optional): Stride length. Defaults to 1.
        padding (str, optional): Padding. Defaults to "same".
        use_bias (bool, optional): Add bias. Defaults to False.
        name (str | None, optional): _description_. Defaults to None.

    Returns:
        tf.keras.layers.Layer: _description_
    """
    name = name + ".conv" if name else None
    return tf.keras.layers.Conv2D(
        filters,
        kernel_size=(1, kernel_size),
        strides=(1, strides),
        padding=padding,
        use_bias=use_bias,
        kernel_initializer=tf.keras.initializers.VarianceScaling(),
        name=name,
    )


def se_block(ratio: int = 8, name: str | None = None) -> Callable[[KerasTensor], KerasTensor]:
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
        y = conv2d(num_chan // ratio, kernel_size=(1, 1), use_bias=True, name=name_sq)(y)
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
        is_downsample = strides != 1 if isinstance(strides, int) else sum(strides) != 1
        add_residual = input_filters == output_filters and not is_downsample
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
        # NOTE: DepthwiseConv2D only supports equal size stride
        # Using Pooling operator for now
        name_dp = f"{name}.dp" if name else None
        y = tf.keras.layers.DepthwiseConv2D(
            kernel_size=kernel_size,
            strides=(1, 1),  # strides,
            padding="same",
            use_bias=False,
            name=name_dp,
        )(y)
        y = batch_norm(name=name_dp)(y)
        y = relu6(name=name_dp)(y)
        if is_downsample:
            y = tf.keras.layers.MaxPool2D(pool_size=strides, padding="same")(y)

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


class SharedWeightsConv(tf.keras.layers.Layer):

    def __init__(
            self,
            parent,
            strides=None,
            padding=None,
            dilation_rate=None,
            activation=None,
            **kwargs
        ):
        conv_classes = (
            tf.keras.layers.Conv1D,
            tf.keras.layers.Conv2D,
            tf.keras.layers.Conv3D
        )
        if not any(isinstance(parent, cls) for cls in conv_classes):
            raise TypeError("'parent' should be a keras convolution layer.")
        super().__init__(**kwargs)
        self.parent = parent
        self.rank = parent.rank
        self.activation = (
            parent.activation if activation is None
            else tf.keras.activations.get(activation)
        )
        cnn_kwargs = {
            'strides': strides,
            'padding': padding,
            'data_format': None,
            'dilation_rate': dilation_rate,
        }
        self.cnn_kwargs = {
            key: getattr(parent, key) if value is None else value
            for key, value in cnn_kwargs.items()
        }
        self.built = self.parent.built
        self.cnn_op = {
            1: tf.keras.backend.conv1d,
            2: tf.keras.backend.conv2d,
            3: tf.keras.backend.conv3d
        }.get(self.rank)

    def build(self, input_shape):
        if not self.built:
            self.parent.build(input_shape)
        self.built = True

    def call(self, inputs):  # adapted from Conv parent layer
        if self.cnn_kwargs['padding'] == 'causal' and self.rank == 1:
            inputs = tf.pad(inputs, self._compute_causal_padding())
        outputs = self.cnn_op(inputs, self.parent.kernel, **self.cnn_kwargs)
        if self.parent.use_bias:
            if self.cnn_kwargs['data_format'] == 'channels_first':
                if self.rank == 1:
                    shape = (1, self.parent.filters, 1)
                    outputs += tf.reshape(self.parent.bias, shape)
                else:
                    outputs = tf.nn.bias_add(
                        outputs, self.parent.bias, data_format='NCHW'
                    )
            else:
                outputs = tf.nn.bias_add(
                    outputs, self.parent.bias, data_format='NHWC'
                )
        if self.activation is not None:
            return self.activation(outputs)
        return outputs

    def _compute_causal_padding(self):  # adapted from Conv parent layer
        left_pad = self.cnn_kwargs['dilation_rate'][0]
        left_pad *= (self.parent.kernel_size[0] - 1)
        if self.cnn_kwargs['data_format'] == 'channels_last':
          causal_padding = [[0, 0], [left_pad, 0], [0, 0]]
        else:
          causal_padding = [[0, 0], [0, 0], [left_pad, 0]]
        return causal_padding



def multiresnet_block(
    d_model: int,
    kernel_size: int | tuple[int, int] = 3,
    depth: int|None = None,
    wavelet_init = None,
    tree_select: str = 'fading',
    seq_len: int|None = None,
    droprate: float = 0,
    memory_size: int|None = None,
    indep_res_init: bool = False,
    name: str | None = None,
) -> Callable[[KerasTensor], KerasTensor]:
    """Multiresnet block

    Args:
        name (str|None, optional): Block name. Defaults to None.

    Returns:
        Callable[[KerasTensor], KerasTensor]: Functional layer

    """

    if depth is None and seq_len is None:
        raise ValueError("Either depth or seq_len must be specified")

    if depth is None:
        depth = math.ceil(math.log2((seq_len - 1) / (kernel_size - 1) + 1))

    if tree_select == 'fading':
        m = depth + 1
    elif memory_size is not None:
        m = memory_size
    else:
        raise ValueError("Either tree_select is fading or memory_size must be specified")

    if wavelet_init is not None:
        import pywt
        wavelet = pywt.Wavelet(wavelet_init)
        h0 = tf.convert_to_tensor(wavelet.dec_lo[::-1])
        h1 = tf.convert_to_tensor(wavelet.dec_hi[::-1])
        h0 = tf.tile(tf.reshape(h0, (1, 1, -1)) , [d_model, 1, 1])
        h1 = tf.tile(tf.reshape(h1, (1, 1, -1)) , [d_model, 1, 1])
    elif kernel_size is not None:
        h0 = 'glorot_uniform'
        h1 = 'glorot_uniform'
    else:
        raise ValueError("Either wavelet_init or kernel_size must be specified")

    def layer(x: KerasTensor) -> KerasTensor:
        if tree_select == 'fading':
            res_lo = x
            y = tf.keras.layers.DepthwiseConv1D(
                kernel_size=1,
            )(x)
            print('IDDQD', x.shape)
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
            for i in range(depth, 0, -1):
                padding = dilation * (kernel_size - 1)
                res_lo_pad = tf.keras.layers.ZeroPadding1D((padding, 0))(res_lo)
                # torch.nn.functional.pad(res_lo, (padding, 0), "constant", 0)
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

        elif tree_select == 'uniform':
            raise NotImplementedError("tree_select == 'uniform' is not implemented yet")

        y = tf.keras.layers.Dropout(droprate)(y)
        y = relu6()(y)
        return y

    return layer
