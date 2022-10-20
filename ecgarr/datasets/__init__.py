import functools
import os
import logging
from typing import Generator, List, Any, Optional, Tuple, Union
import sklearn
import tensorflow as tf
import numpy.typing as npt
import pydantic_argparse
from ..utils import matches_spec, download_file, setup_logger
from ..types import DatasetTypes, EcgTask, EcgDownloadParams
from . import icentia11k

logger = logging.getLogger("ECGARR")


def get_class_names(task: EcgTask) -> List[str]:
    """Get class names for given task

    Args:
        task (EcgTask): Task

    Returns:
        List[str]: class names
    """
    if task == EcgTask.rhythm:
        return ["norm", "afib"]
    if task == EcgTask.beat:
        return ["normal", "pac", "aberrated", "pvc"]
    if task == EcgTask.hr:
        return ["normal", "tachycardia", "bradycardia"]
    raise ValueError(f"unknown task: {task}")


def rhythm_dataset(
    db_path: str,
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
        args=(db_path, patient_ids, frame_size, normalize, samples_per_patient, repeat),
    )
    return dataset


def rhythm_generator(
    db_path: str,
    patient_ids: npt.ArrayLike,
    frame_size: int,
    normalize: bool = True,
    samples_per_patient: Union[int, List[int]] = 1,
    repeat: bool = True,
):
    """Rhythm dataset generator"""
    patient_generator = icentia11k.uniform_patient_generator(
        _str(db_path), patient_ids, repeat=repeat
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
    db_path: str,
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
        args=(db_path, patient_ids, frame_size, normalize, samples_per_patient),
    )
    return dataset


def beat_generator(
    db_path: str,
    patient_ids: npt.ArrayLike,
    frame_size: int,
    normalize: bool = True,
    samples_per_patient: Union[int, List[int]] = 1,
):
    """Beat dataset generator"""
    patient_generator = icentia11k.uniform_patient_generator(
        _str(db_path), patient_ids, repeat=False
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
    db_path: str,
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
        args=(db_path, patient_ids, frame_size, normalize, samples_per_patient),
    )
    return dataset


def heart_rate_generator(
    db_path: str,
    patient_ids: npt.ArrayLike,
    frame_size: int,
    normalize: bool = True,
    samples_per_patient: Union[int, List[int]] = 1,
):
    """Heart rate dataset generator"""
    patient_generator = icentia11k.uniform_patient_generator(
        _str(db_path), patient_ids, repeat=False
    )
    data_generator = icentia11k.heart_rate_data_generator(
        patient_generator,
        frame_size=int(frame_size),
        samples_per_patient=samples_per_patient,
    )
    if normalize:
        data_generator = map(
            lambda x_y: (icentia11k.normalize(x_y[0]), x_y[1]), data_generator
        )
    return data_generator


def _str(s):
    return s.decode() if isinstance(s, bytes) else str(s)


def split_train_test_patients(
    task: EcgTask, patient_ids: npt.ArrayLike, test_size: float
) -> List[List[int]]:
    """Perform train/test split on patients for given task.

    Args:
        task (EcgTask): Arrhythmia task
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
    db_path: str,
    patient_ids: List[int],
    frame_size: int,
    samples_per_patient: Union[int, List[int]] = 1,
    normalize: bool = True,
    repeat: bool = True
):
    """Create dataset from generator for given arrhythmia task

    Args:
        task (EcgTask): Task
        db_path (str): Database path
        patient_ids (List[int]): Patient IDs
        frame_size (int): Frame size
        samples_per_patient (Union[int, List[int]], optional): Samples per patient. Defaults to 1.
        repeat (bool, optional): Generator repeats. Defaults to True.

    Returns:
        Dataset: Patient data generator
    """
    if task == EcgTask.rhythm:
        dataset = rhythm_dataset(
            db_path=db_path,
            patient_ids=patient_ids,
            frame_size=frame_size,
            samples_per_patient=samples_per_patient,
            normalize=normalize,
            repeat=repeat
        )
    elif task == EcgTask.beat:
        dataset = beat_dataset(
            db_path=db_path,
            patient_ids=patient_ids,
            frame_size=frame_size,
            samples_per_patient=samples_per_patient,
            normalize=normalize
        )
    elif task == EcgTask.hr:
        dataset = heart_rate_dataset(
            db_path=db_path,
            patient_ids=patient_ids,
            frame_size=frame_size,
            samples_per_patient=samples_per_patient,
            normalize=normalize
        )
    else:
        raise ValueError(f"unknown task: {task}")
    return dataset

def numpy_dataset_generator(x: npt.ArrayLike, y: npt.ArrayLike) -> Generator[Tuple[npt.ArrayLike, npt.ArrayLike], None, None]:
    """ Create generator from numpy dataset"""
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
    else:
        raise ValueError(f"unknown task: {task}")
    if not matches_spec((x, y), spec, ignore_batch_dim=True):
        raise ValueError(f"data does not match the required spec: {spec}")
    gen = functools.partial(numpy_dataset_generator, x=x, y=y)
    dataset = tf.data.Dataset.from_generator(
        generator=gen,
        output_signature=(
            tf.TensorSpec(shape=(frame_size, 1), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.int32),
        ),
    )
    return dataset


def download_icentia11k_dataset(db_path: str, num_workers: Optional[int] = None, force: bool = False):
    """Download icentia11k dataset

    Args:
        db_path (str): Path to store dataset
        num_workers (Optional[int], optional): # parallel workers. Defaults to None.
        force (bool, optional): Force redownload. Defaults to False.
    """
    logger.info("Downloading icentia11k dataset")
    db_url = (
        "https://physionet.org/static/published-projects/icentia11k-continuous-ecg/"
        "icentia11k-single-lead-continuous-raw-electrocardiogram-dataset-1.0.zip"
    )
    db_zip_path = os.path.join(db_path, "icentia11k.zip")
    os.makedirs(db_path, exist_ok=True)
    if os.path.exists(db_zip_path) and not force:
        logger.warning(
            f"Zip file already exists. Please delete or set `force` flag to redownload. PATH={db_zip_path}"
        )
    else:
        download_file(db_url, db_zip_path, progress=True)

    # 2. Extract and convert patient ECG data to H5 files
    logger.info("Generating icentia11k patient data")
    icentia11k.convert_dataset_zip_to_hdf5(
        zip_path=db_zip_path,
        db_path=db_path,
        force=force,
        num_workers=num_workers
    )
    print("Finished icentia11k patient data")

def download_datasets(params: EcgDownloadParams):
    """Download all specified datasets.

    Args:
        params (EcgDownloadParams): Download parameters
    """
    os.makedirs(params.db_root_path, exist_ok=True)
    #### Icentia11k Dataset
    if "icentia11k" in params.datasets:
        download_icentia11k_dataset(
            db_path=str(params.db_root_path / "icentia11k"),
            num_workers=params.data_parallelism,
            force=params.force
        )

def create_parser():
    """Create CLI parser"""
    return pydantic_argparse.ArgumentParser(
        model=EcgDownloadParams,
        prog="ECG Arrhythmia Dataset",
        description="ECG Arrhythmia dataset",
    )


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_typed_args()
    setup_logger("ECGARR")
    download_datasets(args)
