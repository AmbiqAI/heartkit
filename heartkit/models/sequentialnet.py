"""Simple Sequential Network"""

import keras
import tensorflow as tf
from pydantic import BaseModel, Field

from .blocks import batch_norm, conv2d, relu6, se_block


class SequentialLayerParams(BaseModel):
    """Sequential layer parameters"""

    name: str = Field(..., description="Layer name")
    params: dict = Field(default_factory=dict, description="Layer arguments")


class SequentialNetworkParams(BaseModel):
    """Sequential Network parameters"""

    layers: list[SequentialLayerParams] = Field(default_factory=list, description="Network layers")
    include_top: bool = Field(default=True, description="Include top")
    output_activation: str | None = Field(default=None, description="Output activation")
    model_name: str = Field(default="SequentialNetwork", description="Model name")


def SequentialNetwork(
    x: tf.Tensor,
    params: SequentialNetworkParams,
    num_classes: int | None = None,
) -> keras.Model:
    """Create a simple sequential network"""
    y = x
    for layer in params.layers:
        match layer.name:
            case "conv2d":
                y = conv2d(y, **layer.params)
            case "dense":
                y = keras.layers.Dense(**layer.params)(y)
            case "relu6":
                y = relu6(y)
            case "batch_norm":
                y = batch_norm(y)
            case "se_block":
                y = se_block(y, **layer.params)
            case _:
                raise ValueError(f"Unknown layer {layer.name}")
        # END MATCH
    # END FOR

    if params.include_top:
        if num_classes is not None:
            y = keras.layers.Dense(num_classes)(y)
        if params.output_activation:
            y = keras.layers.Activation(params.output_activation)(y)

    model = keras.Model(x, y, name=params.model_name)
    return model


def sequentialnet_from_object(x: tf.Tensor, params: dict, num_classes: int | None = None) -> keras.Model:
    """Create model from object

    Args:
        x (tf.Tensor): Input tensor
        params (dict): Model parameters.
        num_classes (int, optional): # classes.

    Returns:
        keras.Model: Model
    """
    return SequentialNetwork(x=x, params=SequentialNetworkParams(**params), num_classes=num_classes)
