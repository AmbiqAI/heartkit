import tensorflow as tf
from .features import ecg_feature_extractor
from .resnet1d import ResidualBlock, BottleneckBlock

def build_input_tensor_from_shape(shape, dtype=None, ignore_batch_dim: bool = False):
    """ Build input tensor from shape which can be used to initialize the weights of a model. """
    if isinstance(shape, (list, tuple)):
        return [build_input_tensor_from_shape(
            shape=shape[i],
            dtype=dtype[i] if dtype else None,
            ignore_batch_dim=ignore_batch_dim
        ) for i in range(len(shape))]

    elif isinstance(shape, dict):
        return {k: build_input_tensor_from_shape(
            shape=shape[k],
            dtype=dtype[k] if dtype else None,
            ignore_batch_dim=ignore_batch_dim)
        for k in shape}
    else:
        if ignore_batch_dim:
            shape = shape[1:]
        return tf.keras.layers.Input(shape, dtype=dtype)


def task_solver(task, arch='resnet18', stages=None, return_feature_extractor=False):
    feature_extractor = ecg_feature_extractor(arch, stages=stages)
    last_residual_block = feature_extractor.layers[0].layers[-1]
    if isinstance(last_residual_block, ResidualBlock):
        d_model = last_residual_block.filters
    elif isinstance(last_residual_block, BottleneckBlock):
        d_model = last_residual_block.filters * last_residual_block.expansion
    else:
        raise ValueError('Feature extractor is not a residual network')
    if task == 'rhythm':
        num_classes = 2
        model = tf.keras.Sequential([
            feature_extractor,
            tf.keras.layers.Dense(num_classes),
        ])
    elif task == 'beat':
        num_classes = 5
        model = tf.keras.Sequential([
            feature_extractor,
            tf.keras.layers.Dense(num_classes)
        ])
    elif task == 'hr':
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


def get_pretrained_weights(checkpoint_file, task, arch='resnet18', stages=None):
    model, _ = task_solver(task, arch, stages=stages, return_feature_extractor=True)
    if task in ['rhythm', 'beat', 'hr']:
        inputs = build_input_tensor_from_shape(tf.TensorShape((None, 1)))
    else:
        raise ValueError('unknown task: {}'.format(task))
    model(inputs)
    model.load_weights(checkpoint_file)
    return
