import tensorflow as tf
from keras.engine.keras_tensor import KerasTensor

from .defines import ArchitectureType, HeartTask
from .models.efficientnet import EfficientNetV2, MBConvParam
from .models.features import ecg_feature_extractor


def get_class_names(task: HeartTask) -> list[str]:
    """Get class names for given task

    Args:
        task (HeartTask): Heart task

    Returns:
        list[str]: class names
    """
    if task == HeartTask.rhythm:
        # NOTE: Bucket AFIB and AFL together
        return ["NSR", "AFIB/AFL"]
    if task == HeartTask.beat:
        return ["NORMAL", "PAC", "PVC"]
    if task == HeartTask.hr:
        return ["NORMAL", "TACHYCARDIA", "BRADYCARDIA"]
    if task == HeartTask.segmentation:
        return ["NONE", "P-WAVE", "QRS", "T-WAVE"]
    raise ValueError(f"unknown task: {task}")


def get_num_classes(task: HeartTask) -> int:
    """Get number of classes for given task

    Args:
        task (HeartTask): Heart task

    Returns:
        int: # classes
    """
    return len(get_class_names(task=task))


def get_task_shape(task: HeartTask, frame_size: int) -> tuple[tuple[int], tuple[int]]:
    """Get task model spec

    Args:
        task (HeartTask): Heart task
        frame_size (int): Frame size

    Returns:
        tuple[tf.TensorSpec]: Input shape
    """
    num_classes = get_num_classes(task)
    if task in [HeartTask.hr, HeartTask.rhythm, HeartTask.beat]:
        return (1, frame_size, 1), (num_classes,)
    if task in [HeartTask.segmentation]:
        return (1, frame_size, 1), (frame_size, num_classes)
    raise ValueError(f"unknown task: {task}")


def get_task_spec(task: HeartTask, frame_size: int) -> tuple[tf.TensorSpec, tf.TensorSpec]:
    """Get task model spec

    Args:
        task (HeartTask): Heart task
        frame_size (int): Frame size

    Returns:
        tuple[tf.TensorSpec]: TF spec for task
    """
    in_shape, out_shape = get_task_shape(task, frame_size)
    return (
        tf.TensorSpec(in_shape, tf.float32),
        tf.TensorSpec(out_shape, tf.int32),
    )


def create_task_model(
    inputs: KerasTensor,
    task: HeartTask,
    arch: ArchitectureType = "resnet18",
    stages: int | None = None,
) -> tf.keras.Model:
    """Generate model for given task

    Args:
        inputs (KerasTensor): Model inputs
        task (HeartTask): Heart task
        arch (ArchitectureType, optional): Architecture type. Defaults to 'resnet18'.
        stages (int | None, optional): # stages in network. Defaults to None.

    Returns:
        tf.keras.Model: Model
    """
    num_classes = get_num_classes(task=task)
    if task == HeartTask.segmentation:
        return get_segmentation_model(inputs=inputs, num_classes=num_classes)

    if task == HeartTask.rhythm:
        return get_arrhythmia_model(inputs=inputs, num_classes=num_classes, arch=arch, stages=stages)

    raise NotImplementedError()


def get_arrhythmia_model(
    inputs: KerasTensor,
    num_classes: int,
    arch: ArchitectureType = "resnet18",
    stages: int | None = None,
) -> tf.keras.Model:
    """Default arrhythmia model"""
    blocks = [
        MBConvParam(filters=32, depth=3, ex_ratio=1, kernel_size=(1, 5), strides=(1, 2), se_ratio=2),
        MBConvParam(filters=48, depth=3, ex_ratio=1, kernel_size=(1, 3), strides=(1, 2), se_ratio=4),
        MBConvParam(filters=64, depth=3, ex_ratio=1, kernel_size=(1, 3), strides=(1, 2), se_ratio=4),
        MBConvParam(filters=96, depth=3, ex_ratio=1, kernel_size=(1, 3), strides=(1, 2), se_ratio=4),
    ]
    return EfficientNetV2(
        inputs,
        input_filters=24,
        input_strides=(1, 2),
        input_kernel_size=(1, 7),
        output_filters=0,
        blocks=blocks,
        include_top=True,
        dropout=0.2,
        drop_connect_rate=0.2,
        num_classes=num_classes,
    )


def get_arrhythmia_model_old(
    inputs: KerasTensor,
    num_classes: int,
    arch: ArchitectureType = "resnet18",
    stages: int | None = None,
) -> tf.keras.Model:
    """Old arrhythmia model"""
    x = ecg_feature_extractor(inputs, arch, stages=stages)
    outputs = tf.keras.layers.Dense(num_classes)(x)
    model = tf.keras.Model(inputs, outputs, name="model")
    return model


def get_segmentation_model(inputs: KerasTensor, num_classes: int) -> tf.keras.Model:
    """Load u-net style model.

    Args:
        inputs (KerasTensor): Model input
        num_classes (int): # classes

    Returns:
        tf.keras.Model: Model
    """
    add_skips = True
    filter_lens = [16, 32, 48, 64]
    skip_layers = []

    # Entry block
    name = "STEM"
    x = tf.keras.layers.Conv2D(filter_lens[0], (1, 3), strides=(1, 2), padding="same", name="STEM.CONV1")(inputs)
    x = tf.keras.layers.BatchNormalization(name="STEM.BN1")(x)
    x = tf.keras.layers.Activation("relu", name="STEM.ACT1")(x)
    # if add_skips:
    #     skip_layers.append(x)

    # Downsampling layers
    for i, filters in enumerate(filter_lens[1:]):
        name = f"ENC{i+1}"
        xm = tf.keras.layers.SeparableConv2D(filters, (1, 3), padding="same", name=f"{name}.CONV1")(x)
        xm = tf.keras.layers.BatchNormalization(name=f"{name}.BN1")(xm)
        xm = tf.keras.layers.Activation("relu", name=f"{name}.ACT1")(xm)

        xm = tf.keras.layers.SeparableConv2D(filters, (1, 3), padding="same", name=f"{name}.CONV2")(xm)
        xm = tf.keras.layers.BatchNormalization(name=f"{name}.BN2")(xm)
        xm = tf.keras.layers.Activation("relu", name=f"{name}.ACT2")(xm)
        xm = tf.keras.layers.MaxPooling2D(3, strides=(1, 2), padding="same", name=f"{name}.POOL1")(xm)

        # Project residual
        xr = tf.keras.layers.Conv2D(filters, (1, 1), strides=(1, 2), padding="same", name=f"{name}.CONV3")(x)
        x = tf.keras.layers.add([xm, xr], name=f"{name}.ADD1")
        if add_skips:
            skip_layers.append(x)
    # END FOR

    # Upsampling layers
    for i, filters in enumerate(reversed(filter_lens)):
        name = f"DEC{i+1}"
        xm = tf.keras.layers.Conv2DTranspose(filters, (1, 3), padding="same", name=f"{name}.CONV1")(x)
        xm = tf.keras.layers.BatchNormalization(name=f"{name}.BN1")(xm)
        xm = tf.keras.layers.Activation("relu", name=f"{name}.ACT1")(xm)

        if add_skips and skip_layers:
            skip_layer = skip_layers.pop()
            xm = tf.keras.layers.concatenate([xm, skip_layer])  # Can add or concatenate

        xm = tf.keras.layers.Conv2DTranspose(filters, (1, 3), padding="same", name=f"{name}.CONV2")(xm)
        xm = tf.keras.layers.BatchNormalization(name=f"{name}.BN2")(xm)
        xm = tf.keras.layers.Activation("relu", name=f"{name}.ACT2")(xm)

        xm = tf.keras.layers.UpSampling2D((1, 2), name=f"{name}.UP1")(xm)

        # Project residual
        xr = tf.keras.layers.UpSampling2D((1, 2), name=f"{name}.UP2")(x)
        xr = tf.keras.layers.Conv2D(filters, (1, 1), padding="same", name=f"{name}.CONV3")(xr)
        x = tf.keras.layers.add([xm, xr], name=f"{name}.ADD1")  # Add back residual
    # END FOR

    # Add a per-point classification layer
    x = tf.keras.layers.Conv2D(num_classes, (1, 3), activation=None, padding="same", name="NECK.CONV1")(x)
    outputs = tf.keras.layers.Reshape(x.shape[2:])(x)

    # Define the model
    model = tf.keras.Model(inputs, outputs)
    return model
