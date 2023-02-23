import tensorflow as tf
from tensorflow.python.framework import graph_util, ops
from tensorflow.python.framework.convert_to_constants import (
    convert_variables_to_constants_v2_as_graph,
)
from tensorflow.python.profiler.internal.flops_registry import (
    _binary_per_element_op_flops,
    _reduction_op_flops,
)


@ops.RegisterStatistics("FusedBatchNormV3", "flops")
def _flops_fused_batch_norm_v3(graph, node):
    """inference is only supportted"""
    in_shape = graph_util.tensor_shape_from_node_def_name(graph, node.input[0])
    in_shape.assert_is_fully_defined()
    mean_shape = graph_util.tensor_shape_from_node_def_name(graph, node.input[3])
    mean_shape.assert_is_fully_defined()
    variance_shape = graph_util.tensor_shape_from_node_def_name(graph, node.input[4])
    variance_shape.assert_is_fully_defined()

    if node.attr["is_training"].b is True:
        raise ValueError("Only supports inference mode")

    num_flops = (
        2 * in_shape.num_elements()
        + 5 * variance_shape.num_elements()
        + mean_shape.num_elements()
    )
    return ops.OpStats("flops", num_flops)


@ops.RegisterStatistics("Max", "flops")
def _flops_max(graph, node):
    """inference is supportted"""
    # reduction - comparison, no finalization
    return _reduction_op_flops(graph, node, reduce_flops=1, finalize_flops=0)


@ops.RegisterStatistics("AddV2", "flops")
def _flops_add(graph, node):
    """inference is supportted"""
    return _binary_per_element_op_flops(graph, node)


def get_flops(model: tf.keras.Model, batch_size: int | None = None) -> float:
    """
    Calculate FLOPS for tf.keras.Model or tf.keras.Sequential .
    Ignore operations used in only training mode such as Initialization.
    Use tf.profiler of tensorflow v1 api.
    """
    if not isinstance(model, (tf.keras.Model, tf.keras.Sequential)):
        raise KeyError(
            "model arguments must be tf.keras.Model or tf.keras.Sequential instanse"
        )

    if batch_size is None:
        batch_size = 1

    # convert tf.keras model into frozen graph to count FLOPS about operations used at inference
    # FLOPS depends on batch size
    inputs = [
        tf.TensorSpec([batch_size] + inp.shape[1:], inp.dtype) for inp in model.inputs
    ]
    real_model = tf.function(model).get_concrete_function(inputs)
    frozen_func, _ = convert_variables_to_constants_v2_as_graph(real_model)

    # Calculate FLOPS with tf.profiler
    run_meta = tf.compat.v1.RunMetadata()
    opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
    flops = tf.compat.v1.profiler.profile(
        graph=frozen_func.graph, run_meta=run_meta, cmd="scope", options=opts
    )
    return float(flops.total_float_ops)  # pylint: disable=no-member
