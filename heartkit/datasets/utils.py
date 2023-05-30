import functools
from typing import Generator, TypeVar

import numpy.typing as npt
import tensorflow as tf


def numpy_dataset_generator(
    x: npt.ArrayLike, y: npt.ArrayLike
) -> Generator[tuple[npt.ArrayLike, npt.ArrayLike], None, None]:
    """Create generator from numpy dataset where first axis is samples

    Args:
        x (npt.ArrayLike): X data
        y (npt.ArrayLike): Y data

    Yields:
        Generator[tuple[npt.ArrayLike, npt.ArrayLike], None, None]: Samples
    """
    for i in range(x.shape[0]):
        yield x[i], y[i]


def create_dataset_from_data(
    x: npt.ArrayLike, y: npt.ArrayLike, spec: tuple[tf.TensorSpec]
) -> tf.data.Dataset:
    """Helper function to create dataset from static data
    Args:
        x (npt.ArrayLike): Numpy data
        y (npt.ArrayLike): Numpy labels
    Returns:
        tf.data.Dataset: Dataset
    """
    gen = functools.partial(numpy_dataset_generator, x=x, y=y)
    dataset = tf.data.Dataset.from_generator(generator=gen, output_signature=spec)
    return dataset


T = TypeVar("T")


def buffered_generator(
    generator: Generator[T, None, None], buffer_size: int
) -> Generator[list[T], None, None]:
    """Buffer the elements yielded by a generator. New elements replace the oldest elements in the buffer.

    Args:
        generator (Generator[T]): Generator object.
        buffer_size (int): Number of elements in the buffer.

    Returns:
        Generator[list[T], None, None]: Yields a buffer.
    """
    buffer = []
    for e in generator:
        buffer.append(e)
        if len(buffer) == buffer_size:
            break
    yield buffer
    for e in generator:
        buffer = buffer[1:] + [e]
        yield buffer
