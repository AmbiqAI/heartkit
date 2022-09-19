import os
from typing import List, Any
import sklearn
import tensorflow as tf
import numpy.typing as npt
import pydantic_argparse
from ..utils import matches_spec, download_file, setup_logger
from ..types import EcgTask, EcgDownloadParams
from . import icentia11k

def get_class_names(task: EcgTask) -> List[str]:
    if task == EcgTask.rhythm:
        return ['norm', 'afib']
    if task == EcgTask.beat:
        return ["normal", "pac", "aberrated", "pvc"]
    if task == EcgTask.hr:
        return ["normal", "tachycardia", "bradycardia"]
    raise ValueError('unknown task: {}'.format(task))

def rhythm_dataset(db_path: str, patient_ids: npt.ArrayLike, frame_size: int, normalize: bool = True, samples_per_patient: int = 1, repeat: bool = True):
    dataset = tf.data.Dataset.from_generator(
        generator=rhythm_generator,
        output_types=(tf.float32, tf.int32),
        output_shapes=(tf.TensorShape((frame_size, 1)), tf.TensorShape(())),
        args=(db_path, patient_ids, frame_size, normalize, samples_per_patient, repeat)
    )
    return dataset

def rhythm_generator(db_path: str, patient_ids: npt.ArrayLike, frame_size: int, normalize: bool = True, samples_per_patient: int = 1, repeat: bool = True):
    patient_generator = icentia11k.uniform_patient_generator(_str(db_path), patient_ids, repeat=repeat)
    data_generator = icentia11k.rhythm_data_generator(
        patient_generator, frame_size=int(frame_size), samples_per_patient=int(samples_per_patient)
    )
    if normalize:
        data_generator = map(lambda x_y: (icentia11k.normalize(x_y[0]), x_y[1]), data_generator)
    return data_generator

def beat_dataset(db_path, patient_ids, frame_size, normalize=True, samples_per_patient=1):
    dataset = tf.data.Dataset.from_generator(
        generator=beat_generator,
        output_types=(tf.float32, tf.int32),
        output_shapes=(tf.TensorShape((frame_size, 1)), tf.TensorShape(())),
        args=(db_path, patient_ids, frame_size, normalize, samples_per_patient)
    )
    return dataset

def beat_generator(db_path, patient_ids, frame_size, normalize=True, samples_per_patient=1):
    patient_generator = icentia11k.uniform_patient_generator(_str(db_path), patient_ids, repeat=False)
    data_generator = icentia11k.beat_data_generator(
        patient_generator, frame_size=int(frame_size), samples_per_patient=int(samples_per_patient)
    )
    if normalize:
        data_generator = map(lambda x_y: (icentia11k.normalize(x_y[0]), x_y[1]), data_generator)
    return data_generator

def heart_rate_dataset(db_path, patient_ids, frame_size, normalize=True, samples_per_patient=1):
    dataset = tf.data.Dataset.from_generator(
        generator=heart_rate_generator,
        output_types=(tf.float32, tf.int32),
        output_shapes=(tf.TensorShape((frame_size, 1)), tf.TensorShape(())),
        args=(db_path, patient_ids, frame_size, normalize, samples_per_patient)
    )
    return dataset

def heart_rate_generator(db_path, patient_ids, frame_size, normalize=True, samples_per_patient=1):
    patient_generator = icentia11k.uniform_patient_generator(_str(db_path), patient_ids, repeat=False)
    data_generator = icentia11k.heart_rate_data_generator(
        patient_generator, frame_size=int(frame_size), samples_per_patient=int(samples_per_patient)
    )
    if normalize:
        data_generator = map(lambda x_y: (icentia11k.normalize(x_y[0]), x_y[1]), data_generator)
    return data_generator

def _str(s):
    return s.decode() if isinstance(s, bytes) else str(s)

def split_train_test_patients(task: EcgTask, patient_ids: npt.ArrayLike, test_size: float) -> List[List[Any]]:
    if task == EcgTask.rhythm:
        return icentia11k.train_test_split_patients(patient_ids, test_size=test_size, task=task)
    return sklearn.model_selection.train_test_split(patient_ids, test_size=test_size)

def create_dataset_from_generator(task: EcgTask, db_path: str, patient_ids: List[int], frame_size: int, samples_per_patient: int = 1, repeat: bool = True):
    if task == EcgTask.rhythm:
        dataset = rhythm_dataset(
            db_path=db_path, patient_ids=patient_ids, frame_size=frame_size,
            samples_per_patient=samples_per_patient, repeat=repeat
        )
    elif task == EcgTask.beat:
        dataset = beat_dataset(
            db_path=db_path, patient_ids=patient_ids, frame_size=frame_size,
            samples_per_patient=samples_per_patient
        )
    elif task == EcgTask.hr:
        dataset = heart_rate_dataset(
            db_path=db_path, patient_ids=patient_ids, frame_size=frame_size,
            samples_per_patient=samples_per_patient
        )
    else:
        raise ValueError('unknown task: {}'.format(task))
    return dataset

def create_dataset_from_data(x: npt.ArrayLike, y: npt.ArrayLike, task: EcgTask, frame_size: int):
    if task in [EcgTask.rhythm, EcgTask.beat, EcgTask.hr]:
        spec = (tf.TensorSpec((None, frame_size, 1), tf.float32), tf.TensorSpec((None,), tf.int32))
    else:
        raise ValueError('unknown task: {}'.format(task))
    if not matches_spec((x, y), spec, ignore_batch_dim=True):
        raise ValueError('data does not match the required spec: {}'.format(spec))
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    return dataset

def download_datasets(params: EcgDownloadParams):
    db_zip_path = str(params.db_path / 'icentia11k.zip')
    os.makedirs(params.db_path, exist_ok=True)
    if os.path.exists(db_zip_path) and not params.force:
        print(f'Zip file already exists. Please delete or set `force` flag to redownload. PATH={db_zip_path}')
    else:
        download_file(params.db_url, db_zip_path, progress=True)
    print("#FINISHED downloading dataset")

    #2. Extract and convert patient ECG data to H5 files
    print('#STARTED generating patient data')
    icentia11k.convert_dataset_zip_to_hdf5(
        zip_path=db_zip_path,
        db_path=str(params.db_path),
        force=params.force,
        num_workers=params.data_parallelism
    )
    print('#FINISHED generating patient data')

def create_parser():
    return pydantic_argparse.ArgumentParser(
        model=EcgDownloadParams,
        prog="ECG Arrhythmia Dataset",
        description="ECG Arrhythmia dataset"
    )

if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_typed_args()
    setup_logger('ecgarr', str(args.job_dir))
    download_datasets(args)
