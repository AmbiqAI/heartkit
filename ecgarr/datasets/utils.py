import functools
import logging
from typing import Generator, List, Tuple

import numpy.typing as npt
import tensorflow as tf
from scipy.signal import butter, sosfiltfilt

from ..types import EcgTask

logger = logging.getLogger(__name__)

# if normalize:
#     data_generator = map(
#         lambda x_y: (icentia11k.normalize(x_y[0]), x_y[1]), data_generator
#     )
# if normalize:
#     data_generator = map(
#         lambda x_y: (ludb.normalize(x_y[0]), x_y[1]), data_generator
#     )


def butter_bp_filter(
    data: npt.ArrayLike,
    lowcut: float,
    highcut: float,
    sample_rate: float,
    order: int = 2,
) -> npt.ArrayLike:
    """Apply band-pass filter using butterworth  design and forward-backward cascaded filter

    Args:
        data (npt.ArrayLike): Data
        lowcut (float): Lower cutoff in Hz
        highcut (float): Upper cutoff in Hz
        sample_rate (float): Sampling rate in Hz
        order (int, optional): Filter order. Defaults to 2.

    Returns:
        npt.ArrayLike: Filtered data
    """
    nyq = 0.5 * sample_rate
    low = lowcut / nyq
    high = highcut / nyq
    sos = butter(order, [low, high], btype="band", output="sos")
    f_data = sosfiltfilt(sos, data, axis=0)
    return f_data


def get_class_names(task: EcgTask) -> List[str]:
    """Get class names for given task

    Args:
        task (EcgTask): Task

    Returns:
        List[str]: class names
    """
    if task == EcgTask.rhythm:
        return ["NSR", "AFIB/AFL"]
    if task == EcgTask.beat:
        return ["NORMAL", "PAC", "PVC"]
    if task == EcgTask.hr:
        return ["normal", "tachycardia", "bradycardia"]
    if task == EcgTask.segmentation:
        return ["Other", "P Wave", "QRS", "T Wave"]
    raise ValueError(f"unknown task: {task}")


def get_task_spec(task: EcgTask, frame_size: int) -> Tuple[tf.TensorSpec]:
    """Get task model spec

    Args:
        task (EcgTask): ECG task
        frame_size (int): Frame size

    Returns:
        _type_: _description_
    """
    if task in [EcgTask.rhythm, EcgTask.beat, EcgTask.hr]:
        return (
            tf.TensorSpec((frame_size, 1), tf.float32),
            tf.TensorSpec((), tf.int32),
        )
    if task in [EcgTask.segmentation]:
        return (
            tf.TensorSpec((frame_size, 1), tf.float32),
            tf.TensorSpec((frame_size, 1), tf.int32),
        )
    raise ValueError(f"unknown task: {task}")


def numpy_dataset_generator(
    x: npt.ArrayLike, y: npt.ArrayLike
) -> Generator[Tuple[npt.ArrayLike, npt.ArrayLike], None, None]:
    """Create generator from numpy dataset"""
    for i in range(x.shape[0]):
        yield x[i], y[i]


def create_dataset_from_data(
    x: npt.ArrayLike, y: npt.ArrayLike, task: EcgTask, frame_size: int
):
    """Helper function to create dataset from static data"""
    spec = get_task_spec(task, frame_size)
    gen = functools.partial(numpy_dataset_generator, x=x, y=y)
    dataset = tf.data.Dataset.from_generator(generator=gen, output_signature=spec)
    return dataset
