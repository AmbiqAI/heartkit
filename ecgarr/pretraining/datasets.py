import tensorflow as tf
from ..datasets import icentia11k
import numpy.typing as npt

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
