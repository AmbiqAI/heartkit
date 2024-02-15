import os

import keras
import tensorflow as tf

# pylint: disable=no-name-in-module
from tensorflow.python.profiler.model_analyzer import profile

# pylint: disable=no-name-in-module
from tensorflow.python.profiler.option_builder import ProfileOptionBuilder


def get_flops(model: keras.Model, batch_size: int | None = None, fpath: os.PathLike | None = None) -> float:
    """Calculate FLOPS for keras.Model or keras.Sequential.
    Ignore operations used in only training mode such as Initialization.
    Use tf.profiler of tensorflow v2 api.

    Known Limitations: Does not support LSTM and GRU.

    Args:
        model (keras.Model|keras.Sequential): Model
        batch_size (int, optional): Batch size. Defaults to None.
        fpath (os.PathLike, optional): Output file path. Defaults to None.

    Returns:
        float: FLOPS
    """
    input_signature = [tf.TensorSpec(shape=(batch_size,) + model.input_shape[1:])]
    forward_pass = tf.function(model.call, input_signature=input_signature)
    graph = forward_pass.get_concrete_function().graph
    options = ProfileOptionBuilder.float_operation()
    if fpath:
        options["output"] = f"file:outfile={fpath}"
    graph_info = profile(graph, options=options)
    return float(graph_info.total_float_ops)


class MultiF1Score(keras.metrics.F1Score):
    """Multi-class F1 score"""

    def update_state(self, y_true, y_pred, sample_weight=None):
        if len(y_true.shape) > 2:
            y_true = tf.reshape(y_true, (-1, y_true.shape[-1]))
            y_pred = tf.reshape(y_pred, (-1, y_pred.shape[-1]))

        y_true = tf.cast(y_true, dtype=self.dtype)
        y_pred = tf.cast(y_pred, dtype=self.dtype)
        if not self.built:
            self.build(y_true.shape, y_pred.shape)

        if self.threshold is None:
            threshold = tf.reduce_max(y_pred, axis=-1, keepdims=True)
            # make sure [0, 0, 0] doesn't become [1, 1, 1]
            # Use abs(x) > eps, instead of x != 0 to check for zero
            y_pred = tf.logical_and(y_pred >= threshold, tf.abs(y_pred) > 1e-9)
        else:
            y_pred = y_pred > self.threshold
        y_pred = tf.cast(y_pred, dtype=self.dtype)

        def _weighted_sum(val, sample_weight):
            if sample_weight is not None:
                val = tf.math.multiply(val, tf.expand_dims(sample_weight, 1))
            return tf.reduce_sum(val, axis=self.axis)

        self.true_positives.assign_add(_weighted_sum(y_pred * y_true, sample_weight))
        self.false_positives.assign_add(_weighted_sum(y_pred * (1 - y_true), sample_weight))
        self.false_negatives.assign_add(_weighted_sum((1 - y_pred) * y_true, sample_weight))
        self.intermediate_weights.assign_add(_weighted_sum(y_true, sample_weight))
