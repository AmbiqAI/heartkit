import functools
import logging
import os
from typing import Generator, List, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt
import pydantic_argparse
import sklearn
import tensorflow as tf

from ..types import EcgDownloadParams, EcgTask
from ..utils import load_pkl, matches_spec, save_pkl, setup_logger
from . import icentia11k, ludb

logger = logging.getLogger(__name__)


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


@tf.function
def parallelize_dataset(
    ds_path: str,
    task: EcgTask,
    patient_ids: int = None,
    frame_size: int = 1250,
    samples_per_patient: Union[int, List[int]] = 100,
    repeat: bool = False,
    num_workers: int = 1,
) -> tf.data.Dataset:
    """Generates datasets for given task in parallel using TF `interleave`

    Args:
        ds_path (str): Dataset path
        task (EcgTask, optional): ECG Task routine.
        patient_ids (int, optional): List of patient IDs. Defaults to None.
        frame_size (int, optional): Frame size. Defaults to 1250.
        samples_per_patient (int, optional): # Samples per pateint. Defaults to 100.
        repeat (bool, optional): Should data generator repeat. Defaults to False.
        num_workers (int, optional): Number of parallel workers. Defaults to 1.
    """

    # return create_dataset_from_generator(
    #     task=task,
    #     ds_path=ds_path,
    #     patient_ids=patient_ids,
    #     frame_size=frame_size,
    #     samples_per_patient=samples_per_patient,
    #     repeat=repeat,
    # )

    def _make_train_dataset(i, split):
        return create_dataset_from_generator(
            task=task,
            ds_path=ds_path,
            patient_ids=patient_ids[i * split : (i + 1) * split],
            frame_size=frame_size,
            samples_per_patient=samples_per_patient,
            repeat=repeat,
        )

    split = len(patient_ids) // num_workers
    datasets = [_make_train_dataset(i, split) for i in range(num_workers)]
    par_ds = tf.data.Dataset.from_tensor_slices(datasets)
    return par_ds.interleave(
        lambda x: x,
        cycle_length=num_workers,
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )


def load_datasets(
    ds_path: str,
    task: EcgTask,
    frame_size: int = 1250,
    train_patients: Optional[float] = None,
    val_patients: Optional[float] = None,
    train_pt_samples: Optional[Union[int, List[int]]] = None,
    val_pt_samples: Optional[Union[int, List[int]]] = None,
    val_size: Optional[int] = None,
    val_file: Optional[str] = None,
    num_workers: int = 1,
) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    """Load training and validation TF  datasets
    Args:
        ds_path (str): Dataset path
        task (EcgTask, optional): ECG Heart task.
        frame_size (int, optional): Frame size. Defaults to 1250.
        train_patients (Optional[float], optional): # or proportion of train patients. Defaults to None.
        val_patients (Optional[float], optional): # or proportion of train patients. Defaults to None.
        train_pt_samples (Optional[Union[int, List[int]]], optional): # samples per patient for training. Defaults to None.
        val_pt_samples (Optional[Union[int, List[int]]], optional): # samples per patient for training. Defaults to None.
        train_file (Optional[str], optional): Path to existing pickled training file. Defaults to None.
        val_file (Optional[str], optional): Path to existing pickled validation file. Defaults to None.
        num_workers (int, optional): # of parallel workers. Defaults to 1.

    Returns:
        Tuple[tf.data.Dataset, tf.data.Dataset]: Training and validation datasets
    """

    if val_patients is not None and val_patients >= 1:
        val_patients = int(val_patients)

    train_pt_samples = train_pt_samples or 1000
    if val_pt_samples is None:
        val_pt_samples = train_pt_samples

    # Get train patients
    if task == EcgTask.segmentation:
        train_patient_ids = ludb.get_train_patient_ids()
    else:
        train_patient_ids = icentia11k.get_train_patient_ids()

    if train_patients is not None:
        num_pts = (
            int(train_patients)
            if train_patients > 1
            else int(train_patients * len(train_patient_ids))
        )
        train_patient_ids = train_patient_ids[:num_pts]

    if val_file and os.path.isfile(val_file):
        logger.info(f"Loading validation data from file {val_file}")
        val = load_pkl(val_file)
        val_ds = create_dataset_from_data(
            val["x"], val["y"], task=task, frame_size=frame_size
        )
        val_patient_ids = val["patient_ids"]
        train_patient_ids = np.setdiff1d(train_patient_ids, val_patient_ids)
    else:
        logger.info("Splitting patients into train and validation")
        train_patient_ids, val_patient_ids = split_train_test_patients(
            task=task, patient_ids=train_patient_ids, test_size=val_patients
        )
        if val_size is None:
            val_size = 200 * len(val_patient_ids)

        logger.info(f"Collecting {val_size} validation samples")
        val_ds = parallelize_dataset(
            ds_path=ds_path,
            patient_ids=val_patient_ids,
            task=task,
            frame_size=frame_size,
            samples_per_patient=val_pt_samples,
            repeat=False,
            num_workers=num_workers,
        )
        val_x, val_y = next(val_ds.batch(val_size).as_numpy_iterator())
        val_ds = create_dataset_from_data(
            val_x, val_y, task=task, frame_size=frame_size
        )

        # Cache validation set
        if val_file:
            os.makedirs(os.path.dirname(val_file), exist_ok=True)
            logger.info(f"Caching the validation set in {val_file}")
            save_pkl(val_file, x=val_x, y=val_y, patient_ids=val_patient_ids)
        # END IF
    # END IF

    logger.info("Building train dataset")
    train_ds = parallelize_dataset(
        ds_path=ds_path,
        patient_ids=train_patient_ids,
        task=task,
        frame_size=frame_size,
        samples_per_patient=train_pt_samples,
        repeat=True,
        num_workers=num_workers,
    )
    return train_ds, val_ds


def load_test_dataset(
    ds_path: str,
    task: EcgTask,
    frame_size: int = 1250,
    test_patients: Optional[float] = None,
    test_pt_samples: Optional[Union[int, List[int]]] = None,
    num_workers: int = 1,
) -> tf.data.Dataset:
    """Load testing datasets
    Args:
        ds_path (str): Dataset path
        task (EcgTask, optional): ECG Task. Defaults to EcgTask.rhythm.
        frame_size (int, optional): Frame size. Defaults to 1250.
        train_patients (Optional[float], optional): # or proportion of train patients. Defaults to None.
        val_patients (Optional[float], optional): # or proportion of train patients. Defaults to None.
        train_pt_samples (Optional[int], optional): # samples per patient for training. Defaults to None.
        val_pt_samples (Optional[int], optional): # samples per patient for training. Defaults to None.
        train_file (Optional[str], optional): Path to existing picked training file. Defaults to None.
        val_file (Optional[str], optional): Path to existing picked validation file. Defaults to None.
        num_workers (int, optional): # of parallel workers. Defaults to 1.

    Returns:
        tf.data.Dataset: Test dataset
    """
    if task == EcgTask.segmentation:
        test_patient_ids = ludb.get_test_patient_ids()
    else:
        test_patient_ids = icentia11k.get_test_patient_ids()
    if test_patients is not None:
        num_pts = (
            int(test_patients)
            if test_patients > 1
            else int(test_patients * len(test_patient_ids))
        )
        test_patient_ids = test_patient_ids[:num_pts]
    test_patient_ids = tf.convert_to_tensor(test_patient_ids)
    test_ds = parallelize_dataset(
        ds_path=ds_path,
        patient_ids=test_patient_ids,
        task=task,
        frame_size=frame_size,
        samples_per_patient=test_pt_samples,
        repeat=True,
        num_workers=num_workers,
    )
    return test_ds


def rhythm_dataset(
    ds_path: str,
    patient_ids: npt.ArrayLike,
    frame_size: int,
    normalize: bool = True,
    samples_per_patient: Union[int, List[int]] = 1,
    repeat: bool = True,
):
    """Rhythm dataset"""
    dataset = tf.data.Dataset.from_generator(
        generator=rhythm_generator,
        output_signature=(
            tf.TensorSpec(shape=(frame_size, 1), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.int32),
        ),
        args=(ds_path, patient_ids, frame_size, normalize, samples_per_patient, repeat),
    )
    return dataset


def rhythm_generator(
    ds_path: str,
    patient_ids: npt.ArrayLike,
    frame_size: int,
    normalize: bool = True,
    samples_per_patient: Union[int, List[int]] = 1,
    repeat: bool = True,
):
    """Rhythm dataset generator"""
    patient_generator = icentia11k.uniform_patient_generator(
        _str(ds_path), patient_ids, repeat=repeat
    )
    data_generator = icentia11k.rhythm_data_generator(
        patient_generator,
        frame_size=int(frame_size),
        samples_per_patient=samples_per_patient,
    )
    if normalize:
        data_generator = map(
            lambda x_y: (icentia11k.normalize(x_y[0]), x_y[1]), data_generator
        )
    return data_generator


def beat_dataset(
    ds_path: str,
    patient_ids: npt.ArrayLike,
    frame_size: int,
    normalize: bool = True,
    samples_per_patient: Union[int, List[int]] = 1,
):
    """Beat dataset"""
    dataset = tf.data.Dataset.from_generator(
        generator=beat_generator,
        output_signature=(
            tf.TensorSpec(shape=(frame_size, 1), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.int32),
        ),
        args=(ds_path, patient_ids, frame_size, normalize, samples_per_patient),
    )
    return dataset


def beat_generator(
    ds_path: str,
    patient_ids: npt.ArrayLike,
    frame_size: int,
    normalize: bool = True,
    samples_per_patient: Union[int, List[int]] = 1,
):
    """Beat dataset generator"""
    patient_generator = icentia11k.uniform_patient_generator(
        _str(ds_path), patient_ids, repeat=False
    )
    data_generator = icentia11k.beat_data_generator(
        patient_generator,
        frame_size=int(frame_size),
        samples_per_patient=samples_per_patient,
    )
    if normalize:
        data_generator = map(
            lambda x_y: (icentia11k.normalize(x_y[0]), x_y[1]), data_generator
        )
    return data_generator


def heart_rate_dataset(
    ds_path: str,
    patient_ids: npt.ArrayLike,
    frame_size: int,
    normalize: bool = True,
    samples_per_patient: Union[int, List[int]] = 1,
):
    """Heart rate dataset"""
    dataset = tf.data.Dataset.from_generator(
        generator=heart_rate_generator,
        output_signature=(
            tf.TensorSpec(shape=(frame_size, 1), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.int32),
        ),
        args=(ds_path, patient_ids, frame_size, normalize, samples_per_patient),
    )
    return dataset


def heart_rate_generator(
    ds_path: str,
    patient_ids: npt.ArrayLike,
    frame_size: int,
    normalize: bool = True,
    samples_per_patient: Union[int, List[int]] = 1,
):
    """Heart rate dataset generator"""
    patient_generator = icentia11k.uniform_patient_generator(
        _str(ds_path), patient_ids, repeat=False
    )
    data_generator = icentia11k.heart_rate_data_generator(
        patient_generator,
        frame_size=frame_size,
        samples_per_patient=samples_per_patient,
    )
    if normalize:
        data_generator = map(
            lambda x_y: (icentia11k.normalize(x_y[0]), x_y[1]), data_generator
        )
    return data_generator


def segmentation_dataset(
    ds_path: str,
    patient_ids: npt.ArrayLike,
    frame_size: int,
    normalize: bool = True,
    samples_per_patient: Union[int, List[int]] = 1,
    repeat: bool = True,
):
    """Segmentation dataset"""
    dataset = tf.data.Dataset.from_generator(
        generator=segmentation_generator,
        output_signature=(
            tf.TensorSpec(shape=(frame_size, 1), dtype=tf.float32),
            tf.TensorSpec(shape=(frame_size, 1), dtype=tf.int32),
        ),
        args=(ds_path, patient_ids, frame_size, normalize, samples_per_patient, repeat),
    )
    return dataset


def segmentation_generator(
    ds_path: str,
    patient_ids: npt.ArrayLike,
    frame_size: int,
    normalize: bool = True,
    samples_per_patient: Union[int, List[int]] = 1,
    repeat: bool = True,
):
    """Segmentation dataset generator"""
    patient_generator = ludb.uniform_patient_generator(
        _str(ds_path), patient_ids, repeat=repeat
    )
    data_generator = ludb.segmentation_generator(
        patient_generator,
        frame_size=int(frame_size),
        samples_per_patient=samples_per_patient,
    )
    if normalize:
        data_generator = map(
            lambda x_y: (ludb.normalize(x_y[0]), x_y[1]), data_generator
        )
    return data_generator


def _str(s):
    return s.decode() if isinstance(s, bytes) else str(s)


def split_train_test_patients(
    task: EcgTask, patient_ids: npt.ArrayLike, test_size: float
) -> List[List[int]]:
    """Perform train/test split on patients for given task.

    Args:
        task (EcgTask): Heart task
        patient_ids (npt.ArrayLike): Patient Ids
        test_size (float): Test size

    Returns:
        List[List[int]]: Train and test sets of patient ids
    """
    if task == EcgTask.rhythm:
        return icentia11k.train_test_split_patients(
            patient_ids, test_size=test_size, task=task
        )
    return sklearn.model_selection.train_test_split(patient_ids, test_size=test_size)


def create_dataset_from_generator(
    task: EcgTask,
    ds_path: str,
    patient_ids: List[int],
    frame_size: int,
    samples_per_patient: Union[int, List[int]] = 1,
    normalize: bool = True,
    repeat: bool = True,
):
    """Create dataset from generator for given task

    Args:
        task (EcgTask): Task
        ds_path (str): Dataset path
        patient_ids (List[int]): Patient IDs
        frame_size (int): Frame size
        samples_per_patient (Union[int, List[int]], optional): Samples per patient. Defaults to 1.
        repeat (bool, optional): Generator repeats. Defaults to True.

    Returns:
        Dataset: Patient data generator
    """
    if task == EcgTask.rhythm:
        dataset = rhythm_dataset(
            ds_path=ds_path,
            patient_ids=patient_ids,
            frame_size=frame_size,
            samples_per_patient=samples_per_patient,
            normalize=normalize,
            repeat=repeat,
        )
    elif task == EcgTask.beat:
        dataset = beat_dataset(
            ds_path=ds_path,
            patient_ids=patient_ids,
            frame_size=frame_size,
            samples_per_patient=samples_per_patient,
            normalize=normalize,
        )
    elif task == EcgTask.hr:
        dataset = heart_rate_dataset(
            ds_path=ds_path,
            patient_ids=patient_ids,
            frame_size=frame_size,
            samples_per_patient=samples_per_patient,
            normalize=normalize,
        )
    elif task == EcgTask.segmentation:
        dataset = segmentation_dataset(
            ds_path=ds_path,
            patient_ids=patient_ids,
            frame_size=frame_size,
            samples_per_patient=samples_per_patient,
            normalize=normalize,
        )

    else:
        raise ValueError(f"unknown task: {task}")
    return dataset


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
    if task in [EcgTask.rhythm, EcgTask.beat, EcgTask.hr]:
        spec = (
            tf.TensorSpec((None, frame_size, 1), tf.float32),
            tf.TensorSpec((None,), tf.int32),
        )
    elif task in [EcgTask.segmentation]:
        spec = (
            tf.TensorSpec((None, frame_size, 1), tf.float32),
            tf.TensorSpec((None, frame_size, 1), tf.int32),
        )
    else:
        raise ValueError(f"unknown task: {task}")
    if not matches_spec((x, y), spec, ignore_batch_dim=True):
        raise ValueError(f"data does not match the required spec: {spec}")
    gen = functools.partial(numpy_dataset_generator, x=x, y=y)
    dataset = tf.data.Dataset.from_generator(
        generator=gen,
        # output_signature=spec
        output_signature=(
            tf.TensorSpec(shape=(frame_size, 1), dtype=tf.float32),
            tf.TensorSpec(shape=(frame_size, 1), dtype=tf.int32),
        ),
    )
    return dataset


def download_icentia11k_dataset(
    ds_path: str, num_workers: Optional[int] = None, force: bool = False
):
    """Download icentia11k dataset

    Args:
        ds_path (str): Path to store dataset
        num_workers (Optional[int], optional): # parallel workers. Defaults to None.
        force (bool, optional): Force redownload. Defaults to False.
    """
    icentia11k.download_dataset(ds_path, num_workers=num_workers, force=force)


def download_ludb_dataset(
    ds_path: str, num_workers: Optional[int] = None, force: bool = False
):
    """Download LUDB dataset

    Args:
        ds_path (str): Path to store dataset
        num_workers (Optional[int], optional): # parallel workers. Defaults to None.
        force (bool, optional): Force redownload. Defaults to False.
    """
    ludb.download_dataset(ds_path=ds_path, num_workers=num_workers, force=force)


def download_datasets(params: EcgDownloadParams):
    """Download all specified datasets.

    Args:
        params (EcgDownloadParams): Download parameters
    """
    os.makedirs(params.db_root_path, exist_ok=True)
    #### Icentia11k Dataset
    if "icentia11k" in params.datasets:
        download_icentia11k_dataset(
            ds_path=str(params.db_root_path / "icentia11k"),
            num_workers=params.data_parallelism,
            force=params.force,
        )
    if "ludb" in params.datasets:
        download_ludb_dataset(
            ds_path=str(params.db_root_path / "ludb"),
            num_workers=params.data_parallelism,
            force=params.force,
        )


def create_parser():
    """Create CLI parser"""
    return pydantic_argparse.ArgumentParser(
        model=EcgDownloadParams,
        prog="ECG Dataset",
        description="ECG dataset",
    )


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_typed_args()
    setup_logger("ECGARR")
    download_datasets(args)
