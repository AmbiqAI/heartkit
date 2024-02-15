"""Implementation of TSMixer."""

from typing import Literal

import keras
import tensorflow as tf
from pydantic import BaseModel, Field

from .defines import KerasLayer


class TsBlockParams(BaseModel):
    """TsMixer block parameters"""

    norm: Literal["batch", "layer"] | None = Field(default="layer", description="Normalization type")
    activation: Literal["relu", "gelu"] | None = Field(default="relu", description="Activation type")
    dropout: float | None = Field(default=None, description="Dropout rate")
    ff_dim: int | None = Field(default=None, description="Feed forward dimension")


class TsMixerParams(BaseModel):
    """TsMixer parameters"""

    blocks: list[TsBlockParams] = Field(default_factory=list, description="UNext blocks")
    model_name: str = Field(default="TsMixer", description="Model name")


def norm_layer(norm: str, name: str) -> KerasLayer:
    """Normalization layer

    Args:
        norm (str): Normalization type
        name (str): Name

    Returns:
        KerasLayer: Layer
    """

    def layer(x: tf.Tensor) -> tf.Tensor:
        """Functional normalization layer

        Args:
            x (tf.Tensor): Input tensor

        Returns:
            tf.Tensor: Output tensor
        """
        if norm == "batch":
            return keras.layers.BatchNormalization(axis=[-2, -1], name=f"{name}.BN")(x)
        if norm == "layer":
            return keras.layers.LayerNormalization(axis=[-2, -1], name=f"{name}.LN")(x)
        return x

    return layer


def ts_block(params: TsBlockParams, name: str) -> KerasLayer:
    """Residual block of TSMixer."""

    def layer(x: tf.Tensor) -> tf.Tensor:
        # Temporal Linear
        y = norm_layer(params.norm, name=f"{name}.TL")(x)
        y = tf.transpose(y, perm=[0, 2, 1])  # [Batch, Channel, Input Length]
        y = keras.layers.Dense(y.shape[-1], activation=params.activation, name=f"{name}.TL.DENSE")(y)
        y = tf.transpose(y, perm=[0, 2, 1])  # [Batch, Input Length, Channel]
        y = keras.layers.Dropout(params.dropout, name=f"{name}.TL.DROP")(y)
        res = y + x

        # Feature Linear
        y = norm_layer(params.norm, name=f"{name}.FL")(res)
        y = keras.layers.Dense(params.ff_dim, activation=params.activation, name=f"{name}.FL.DENSE")(
            y
        )  # [Batch, Input Length, FF_Dim]
        y = keras.layers.Dropout(params.dropout, name=f"{name}.FL.DROP")(y)

        y = keras.layers.Dense(x.shape[-1], name=f"{name}.RL.DENSE")(y)  # [Batch, Input Length, Channel]
        y = keras.layers.Dropout(params.dropout, name=f"{name}.RL.DROP")(y)
        return y + res

    return layer


def TsMixer(x: tf.Tensor, params: TsMixerParams, num_classes: int):
    """Create TSMixer TF functional model

    Args:
        x (tf.Tensor): Input tensor
        params (TsMixerParams): Parameters
        num_classes (int): Number of classes
    """
    y = x
    for i, block in enumerate(params.blocks):
        y = ts_block(block, name=f"B{i}")(y)

    # if target_slice:
    #     y = y[:, :, target_slice]

    y = tf.transpose(y, perm=[0, 2, 1])  # [Batch, Channel, Input Length]
    y = keras.layers.Dense(num_classes)(y)  # [Batch, Channel, Output Length]
    y = tf.transpose(y, perm=[0, 2, 1])  # [Batch, Output Length, Channel])

    # Define the model
    model = keras.Model(x, y, name=params.model_name)
    return model
