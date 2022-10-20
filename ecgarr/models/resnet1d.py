from typing import Tuple
import tensorflow as tf
from keras.engine.keras_tensor import KerasTensor


def batch_norm() -> tf.keras.layers.Layer:
    """Batch normalization layer"""
    return tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5)


def relu() -> tf.keras.layers.Layer:
    "ReLU layer"
    return tf.keras.layers.ReLU()


def conv12d(
    filters: int, kernel_size: int = 3, strides: int = 1
) -> tf.keras.layers.Layer:
    """1D convolutional layer using 2D convolutional layer"""
    return tf.keras.layers.Conv2D(
        filters,
        kernel_size=(1, kernel_size),
        strides=(1, strides),
        padding="same",
        use_bias=False,
        kernel_initializer=tf.keras.initializers.VarianceScaling(),
    )


def conv1d(
    filters: int, kernel_size: int = 3, strides: int = 1
) -> tf.keras.layers.Layer:
    """1D convolutional layer"""
    return tf.keras.layers.Conv1D(
        filters,
        kernel_size,
        strides=strides,
        padding="same",
        use_bias=False,
        kernel_initializer=tf.keras.initializers.VarianceScaling(),
    )


def generate_bottleneck_block(
    x: KerasTensor,
    filters: int,
    kernel_size: int = 3,
    strides: int = 1,
    expansion: int = 4,
) -> KerasTensor:
    """Generate functional bottleneck block.

    Args:
        x (KerasTensor): Input
        filters (int): Filter size
        kernel_size (int, optional): Kernel size. Defaults to 3.
        strides (int, optional): Stride length. Defaults to 1.
        expansion (int, optional): Expansion factor. Defaults to 4.

    Returns:
        KerasTensor: Outputs
    """
    num_chan = x.shape[-1]
    projection = num_chan != filters * expansion or strides > 1

    bx = conv12d(filters, 1, 1)(x)
    bx = batch_norm()(bx)
    bx = relu()(bx)

    bx = conv12d(filters, kernel_size, strides)(x)
    bx = batch_norm()(bx)
    bx = relu()(bx)

    bx = conv12d(filters * expansion, 1, 1)(bx)
    bx = batch_norm()(bx)

    if projection:
        x = conv12d(filters * expansion, 1, strides)(x)
        x = batch_norm()(x)
    x = tf.keras.layers.add([bx, x])
    x = relu()(x)
    return x


def generate_residual_block(
    x: KerasTensor,
    filters: int,
    kernel_size: int = 3,
    strides: int = 1,
) -> KerasTensor:
    """Generate functional residual block

    Args:
        x (KerasTensor): Input
        filters (int): Filter size
        kernel_size (int, optional): Kernel size. Defaults to 3.
        strides (int, optional): Stride length. Defaults to 1.

    Returns:
        KerasTensor: Outputs
    """

    num_chan = x.shape[-1]
    projection = num_chan != filters or strides > 1

    bx = conv12d(filters, kernel_size, strides)(x)
    bx = batch_norm()(bx)
    bx = relu()(bx)

    bx = conv12d(filters, kernel_size, 1)(bx)
    bx = batch_norm()(bx)
    if projection:
        x = conv12d(filters, 1, strides)(x)
        x = batch_norm()(x)
    x = tf.keras.layers.add([bx, x])
    x = relu()(x)
    return x


def generate_resnet(
    inputs: KerasTensor,
    num_outputs: int = 1,
    blocks: Tuple[int, ...] = (2, 2, 2, 2),
    filters: Tuple[int, ...] = (64, 128, 256, 512),
    kernel_size: Tuple[int, ...] = (3, 3, 3, 3),
    input_conv: Tuple[int, ...] = (64, 7, 2),
    use_bottleneck: bool = False,
    include_top: bool = True,
) -> KerasTensor:
    """Generate functional 1D ResNet model.
    NOTE: We leverage functional model design as well as 2D architecture to enable QAT.
          For TFL and TFLM, they automatically convert 1D layers to 2D, however,
          TF doesnt do this for QAT and instead throws an error...
    Args:
        inputs (KerasTensor): Inputs
        num_outputs (int, optional): # class outputs. Defaults to 1.
        blocks (Tuple[int, ...], optional): Stage block sizes. Defaults to (2, 2, 2, 2).
        filters (Tuple[int, ...], optional): Stage filter sizes. Defaults to (64, 128, 256, 512).
        kernel_size (Tuple[int, ...], optional): Stage kernel sizes. Defaults to (3, 3, 3, 3).
        input_conv (Tuple[int, ...], optional): Initial conv layer attributes. Defaults to (64, 7, 2).
        include_top (bool, optional): Include classifier layers. Defaults to True.

    Returns:
        KerasTensor: Outputs
    """
    x = tf.keras.layers.Reshape([1] + inputs.shape[1:])(inputs)
    x = conv12d(*input_conv)(x)
    x = batch_norm()(x)
    x = relu()(x)
    x = tf.keras.layers.MaxPooling2D(3, (1, 2), padding="same")(x)
    for stage, num_blocks in enumerate(blocks):
        for block in range(num_blocks):
            strides = 2 if block == 0 and stage > 0 else 1
            if use_bottleneck:
                x = generate_bottleneck_block(
                    x=x,
                    filters=filters[stage],
                    kernel_size=kernel_size[stage],
                    strides=strides,
                )
            else:
                x = generate_residual_block(
                    x=x,
                    filters=filters[stage],
                    kernel_size=kernel_size[stage],
                    strides=strides,
                )
        # END FOR
    # END FOR
    x = tf.keras.layers.Reshape(x.shape[2:])(x)
    if include_top:
        out_act = "sigmoid" if num_outputs == 1 else "softmax"
        x = tf.keras.layers.GlobalAveragePooling1D()(x)
        x = tf.keras.layers.Dense(num_outputs, out_act)(x)
    return x
