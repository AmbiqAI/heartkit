import keras
import tensorflow as tf

from .defines import KerasLayer


def layer_norm(name: str | None = None, axis=-1, scale: bool = True) -> KerasLayer:
    """Layer normalization layer"""
    name = name + ".ln" if name else None
    return keras.layers.LayerNormalization(axis=axis, name=name, scale=scale)


def batch_norm(name: str | None = None, momentum=0.9, epsilon=1e-3) -> KerasLayer:
    """Batch normalization layer"""
    name = name + ".bn" if name else None
    return keras.layers.BatchNormalization(momentum=momentum, epsilon=epsilon, name=name)


def glu(dim: int = -1) -> KerasLayer:
    """Gated linear unit layer"""

    def layer(x: tf.Tensor) -> tf.Tensor:
        out, gate = tf.split(x, num_or_size_splits=2, axis=dim)
        gate = tf.sigmoid(gate)
        x = tf.multiply(out, gate)
        return x

    # END DEF
    return layer


def relu(name: str | None = None) -> KerasLayer:
    """ReLU activation layer"""
    name = name + ".act" if name else None
    return keras.layers.ReLU(name=name)


def relu6(name: str | None = None) -> KerasLayer:
    """Hard ReLU activation layer"""
    name = name + ".act" if name else None
    return keras.layers.Activation(tf.nn.relu6, name=name)


def mish(name: str | None = None) -> KerasLayer:
    """Mish activation layer"""
    name = name + ".act" if name else None
    return keras.layers.Activation(keras.activations.mish, name=name)


def gelu(name: str | None = None) -> KerasLayer:
    """GeLU activation layer"""
    name = name + ".act" if name else None
    return keras.layers.Activation("gelu", name=name)


def hard_sigmoid(name: str | None = None) -> KerasLayer:
    """Hard sigmoid activation layer"""
    name = name + ".act" if name else None
    return keras.layers.Activation(keras.activations.hard_sigmoid, name=name)


def conv2d(
    filters: int,
    kernel_size: int | tuple[int, int] = 3,
    strides: int | tuple[int, int] = 1,
    padding: str = "same",
    use_bias: bool = False,
    groups: int = 1,
    dilation: int = 1,
    name: str | None = None,
) -> KerasLayer:
    """2D convolutional layer

    Args:
        filters (int): # filters
        kernel_size (int | tuple[int, int], optional): Kernel size. Defaults to 3.
        strides (int | tuple[int, int], optional): Stride length. Defaults to 1.
        padding (str, optional): Padding. Defaults to "same".
        use_bias (bool, optional): Add bias. Defaults to False.
        name (str | None, optional): Layer name. Defaults to None.

    Returns:
        KerasLayer: Functional 2D conv layer
    """
    name = name + ".conv" if name else None
    return keras.layers.Conv2D(
        filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        use_bias=use_bias,
        groups=groups,
        dilation_rate=dilation,
        kernel_initializer="he_normal",
        name=name,
    )


def conv1d(
    filters: int,
    kernel_size: int = 3,
    strides: int = 1,
    padding: str = "same",
    use_bias: bool = False,
    name: str | None = None,
) -> KerasLayer:
    """1D convolutional layer using 2D convolutional layer

    Args:
        filters (int): # filters
        kernel_size (int, optional): Kernel size. Defaults to 3.
        strides (int, optional): Stride length. Defaults to 1.
        padding (str, optional): Padding. Defaults to "same".
        use_bias (bool, optional): Add bias. Defaults to False.
        name (str | None, optional): Layer name. Defaults to None.

    Returns:
        KerasLayer: Functional 1D conv layer
    """
    name = name + ".conv" if name else None
    return keras.layers.Conv2D(
        filters,
        kernel_size=(1, kernel_size),
        strides=(1, strides),
        padding=padding,
        use_bias=use_bias,
        kernel_initializer="he_normal",
        name=name,
    )


def se_block(ratio: int = 8, name: str | None = None) -> KerasLayer:
    """Squeeze & excite block

    Args:
        ratio (Expansion ratio, optional): Expansion ratio. Defaults to 8.
        name (str|None, optional): Block name. Defaults to None.

    Returns:
        KerasLayer: Functional SE layer
    """

    def layer(x: tf.Tensor) -> tf.Tensor:
        num_chan = x.shape[-1]
        # Squeeze
        name_pool = f"{name}.pool" if name else None
        name_sq = f"{name}.sq" if name else None
        y = keras.layers.GlobalAveragePooling2D(name=name_pool, keepdims=True)(x)
        y = conv2d(num_chan // ratio, kernel_size=(1, 1), use_bias=True, name=name_sq)(y)
        y = relu6(name=name_sq)(y)
        # Excite
        name_ex = f"{name}.ex" if name else None
        y = conv2d(num_chan, kernel_size=(1, 1), use_bias=True, name=name_ex)(y)
        y = hard_sigmoid(name=name_ex)(y)
        y = keras.layers.Multiply()([x, y])
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
) -> KerasLayer:
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
        KerasLayer: Functional layer
    """

    def layer(x: tf.Tensor) -> tf.Tensor:
        input_filters = x.shape[-1]
        stride_len = strides if isinstance(strides, int) else sum(strides) / len(strides)
        is_downsample = stride_len > 1
        add_residual = input_filters == output_filters and not is_downsample
        # Expand: narrow -> wide
        if expand_ratio != 1:
            name_ex = f"{name}.exp" if name else None
            filters = int(input_filters * expand_ratio)
            y = conv2d(filters, kernel_size=(1, 1), strides=(1, 1), name=name_ex)(x)
            y = batch_norm(name=name_ex)(y)
            y = relu6(name=name_ex)(y)
        else:
            y = x

        # Apply: wide -> wide
        # NOTE: DepthwiseConv2D only supports equal size stride -> use maxpooling instead
        name_dp = f"{name}.dp" if name else None
        y = keras.layers.DepthwiseConv2D(
            kernel_size=kernel_size,
            strides=(1, 1),
            padding="same",
            use_bias=False,
            depthwise_initializer="he_normal",
            name=name_dp,
        )(y)
        y = batch_norm(name=name_dp)(y)
        y = relu6(name=name_dp)(y)
        if is_downsample:
            y = keras.layers.MaxPool2D(pool_size=strides, padding="same")(y)

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
                y = keras.layers.Dropout(droprate, noise_shape=(None, 1, 1, 1))(y)
            y = keras.layers.add([x, y], name=name_res)
        return y

    return layer


class SharedWeightsConv(keras.layers.Layer):
    """Allows sharing weights between conv layers."""

    def __init__(
        self,
        parent,
        strides=None,
        padding=None,
        dilation_rate=None,
        activation=None,
        **kwargs,
    ):
        conv_classes = (
            keras.layers.Conv1D,
            keras.layers.Conv2D,
            keras.layers.Conv3D,
        )
        if not any(isinstance(parent, cls) for cls in conv_classes):
            raise TypeError("'parent' should be a keras convolution layer.")
        super().__init__(**kwargs)
        self.parent = parent
        self.rank = parent.rank
        self.activation = parent.activation if activation is None else keras.activations.get(activation)
        cnn_kwargs = {
            "strides": strides,
            "padding": padding,
            "data_format": None,
            "dilation_rate": dilation_rate,
        }
        self.cnn_kwargs = {key: getattr(parent, key) if value is None else value for key, value in cnn_kwargs.items()}
        self.built = self.parent.built
        self.cnn_op = {
            1: keras.backend.conv1d,
            2: keras.backend.conv2d,
            3: keras.backend.conv3d,
        }.get(self.rank)

    def build(self, input_shape):
        if not self.built:
            self.parent.build(input_shape)
        self.built = True

    def call(self, inputs, *args, **kwargs):  # adapted from Conv parent layer
        if self.cnn_kwargs["padding"] == "causal" and self.rank == 1:
            inputs = tf.pad(inputs, self._compute_causal_padding())
        outputs = self.cnn_op(inputs, self.parent.kernel, **self.cnn_kwargs)
        if self.parent.use_bias:
            if self.cnn_kwargs["data_format"] == "channels_first":
                if self.rank == 1:
                    shape = (1, self.parent.filters, 1)
                    outputs += tf.reshape(self.parent.bias, shape)
                else:
                    outputs = tf.nn.bias_add(outputs, self.parent.bias, data_format="NCHW")
            else:
                outputs = tf.nn.bias_add(outputs, self.parent.bias, data_format="NHWC")
        if self.activation is not None:
            return self.activation(outputs)
        return outputs

    def _compute_causal_padding(self):  # adapted from Conv parent layer
        left_pad = self.cnn_kwargs["dilation_rate"][0]
        left_pad *= self.parent.kernel_size[0] - 1
        if self.cnn_kwargs["data_format"] == "channels_last":
            causal_padding = [[0, 0], [left_pad, 0], [0, 0]]
        else:
            causal_padding = [[0, 0], [0, 0], [left_pad, 0]]
        return causal_padding
