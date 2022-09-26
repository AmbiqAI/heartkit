from typing import Any, Dict, List, Optional, Tuple, Union
import tensorflow as tf
from .features import ecg_feature_extractor
from .resnet1d import ResidualBlock, BottleneckBlock
from ..types import EcgTask, ArchitectureType

InputShape = Union[Tuple[int], List[Tuple[int]], Dict[str, Tuple[int]]]

def build_input_tensor_from_shape(shape: InputShape, dtype: tf.DType = None, ignore_batch_dim: bool = False):
    """ Build input tensor from shape which can be used to initialize the weights of a model.

    Args:
        shape (InputShape]): Input Shape
        dtype (tf.DType, optional): _description_. Defaults to None.
        ignore_batch_dim (bool, optional): Ignore first dimension as batch. Defaults to False.

    Returns:
        _type_: _description_
    """
    if isinstance(shape, (list, tuple)):
        return [build_input_tensor_from_shape(
            shape=shape[i],
            dtype=dtype[i] if dtype else None,
            ignore_batch_dim=ignore_batch_dim
        ) for i in range(len(shape))]

    if isinstance(shape, dict):
        return {k: build_input_tensor_from_shape(
            shape=shape[k],
            dtype=dtype[k] if dtype else None,
            ignore_batch_dim=ignore_batch_dim)
        for k in shape}

    if ignore_batch_dim:
        shape = shape[1:]
    return tf.keras.layers.Input(shape, dtype=dtype)

def unfold_model_layers(layer, model: Optional[tf.keras.Model] = None):
    """ Unfold model layers into flat """
    if model is None:
        model = tf.keras.Sequential()
    if isinstance(layer, (tf.keras.Model, tf.keras.Sequential)):
        print(f'Unfolding {layer}')
        for llayer in layer.layers:
            unfold_model_layers(llayer, model)
    else:
        print(f'Adding layer {layer}')
        model.add(layer)
    return model


def task_solver(
        task: EcgTask,
        arch: ArchitectureType = 'resnet18',
        stages: Optional[int] = None,
        return_feature_extractor: bool = False
    ) -> Union[Tuple[tf.keras.Model, tf.keras.Sequential], tf.keras.Model]:
    """ Generate model for given arrhythmia task

    Args:
        task (EcgTask): Heart arrhythmia task
        arch (ArchitectureType, optional): Architecture type. Defaults to 'resnet18'.
        stages (Optional[int], optional): # stages in network. Defaults to None.
        return_feature_extractor (bool, optional): Include feature extractor. Defaults to False.

    Returns:
        Union[Tuple[tf.keras.Model, tf.keras.Sequential], tf.keras.Model]: Model and optional feature extractor stage
    """

    feature_extractor = ecg_feature_extractor(arch, stages=stages)
    last_residual_block = feature_extractor.layers[0].layers[-1]
    if isinstance(last_residual_block, ResidualBlock):
        d_model = last_residual_block.filters
    elif isinstance(last_residual_block, BottleneckBlock):
        d_model = last_residual_block.filters * last_residual_block.expansion
    else:
        raise ValueError('Feature extractor is not a residual network')
    if task == EcgTask.rhythm:
        num_classes = 2
        model = tf.keras.Sequential([
            feature_extractor,
            tf.keras.layers.Dense(num_classes),
        ])
    elif task == EcgTask.beat:
        num_classes = 5
        model = tf.keras.Sequential([
            feature_extractor,
            tf.keras.layers.Dense(num_classes)
        ])
    elif task == EcgTask.hr:
        num_classes = 4
        model = tf.keras.Sequential([
            feature_extractor,
            tf.keras.layers.Dense(num_classes)
        ])
    else:
        raise ValueError('unknown task: {}'.format(task))

    if return_feature_extractor:
        return model, feature_extractor
    else:
        return model

def get_pretrained_weights(
        checkpoint_file: str,
        task: EcgTask,
        arch: ArchitectureType = 'resnet18',
        stages: Optional[int] = None
    ) -> tf.keras.Model:
    """ Initialize model with weights from file

    Args:
        checkpoint_file (str): TensorFlow checkpoint file containing weights
        task (EcgTask): Hear arrhythmia task
        arch (ArchitectureType, optional): Architecture type. Defaults to 'resnet18'.
        stages (Optional[int], optional): # stages in network. Defaults to None.

    Returns:
        tf.keras.Model: Pre-trained model
    """
    model, _ = task_solver(task, arch, stages=stages, return_feature_extractor=True)
    if task in [EcgTask.rhythm, EcgTask.beat, EcgTask.hr]:
        inputs = build_input_tensor_from_shape(tf.TensorShape((None, 1)))
    else:
        raise ValueError('Unknown task: {}'.format(task))
    model(inputs)
    model.load_weights(checkpoint_file)
    return model
