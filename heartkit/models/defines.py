from typing import Callable

import tensorflow as tf
from pydantic import BaseModel, Field

KerasLayer = Callable[[tf.Tensor], tf.Tensor]


class MBConvParams(BaseModel):
    """MBConv parameters"""

    filters: int = Field(..., description="# filters")
    depth: int = Field(default=1, description="Layer depth")
    ex_ratio: float = Field(default=1, description="Expansion ratio")
    kernel_size: int | tuple[int, int] = Field(default=3, description="Kernel size")
    strides: int | tuple[int, int] = Field(default=1, description="Stride size")
    se_ratio: float = Field(default=8, description="Squeeze Excite ratio")
    droprate: float = Field(default=0, description="Drop rate")
