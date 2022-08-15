import tensorflow as tf
from ..datasets import icentia11k
from ..utils import buffered_generator

def rhythm_dataset(db_path, patient_ids, frame_size, normalize=True, samples_per_patient=1):
    dataset = tf.data.Dataset.from_generator(
        generator=rhythm_generator,
        output_types=(tf.float32, tf.int32),
        output_shapes=(tf.TensorShape((frame_size, 1)), tf.TensorShape(())),
        args=(db_path, patient_ids, frame_size, normalize, samples_per_patient)
    )
    return dataset


def rhythm_generator(db_path, patient_ids, frame_size, normalize=True, samples_per_patient=1):
    patient_generator = icentia11k.uniform_patient_generator(_str(db_path), patient_ids)
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
    patient_generator = icentia11k.uniform_patient_generator(_str(db_path), patient_ids)
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
    patient_generator = icentia11k.uniform_patient_generator(_str(db_path), patient_ids)
    data_generator = icentia11k.heart_rate_data_generator(
        patient_generator, frame_size=int(frame_size), samples_per_patient=int(samples_per_patient)
    )
    if normalize:
        data_generator = map(lambda x_y: (icentia11k.normalize(x_y[0]), x_y[1]), data_generator)
    return data_generator


def cpc_dataset(
        db_path, patient_ids, frame_size, context_size, ns, context_overlap=0,
        positive_offset=0, num_buffered_patients=16, normalize=True,
        samples_per_patient=1
    ):
    dataset = tf.data.Dataset.from_generator(
        generator=cpc_generator,
        output_types=({'context': tf.float32, 'samples': tf.float32}, tf.int32),
        output_shapes=({
            'context': tf.TensorShape((context_size, frame_size, 1)),
            'samples': tf.TensorShape((ns + 1, frame_size, 1))
        }, tf.TensorShape(())),
        args=(
            db_path, patient_ids, frame_size, context_size, ns, context_overlap,
            positive_offset, num_buffered_patients, normalize, samples_per_patient
        )
    )
    return dataset


def cpc_generator(
        db_dir, patient_ids, frame_size, context_size, ns, context_overlap=0,
        positive_offset=0, num_buffered_patients=16, normalize=True, samples_per_patient=1
    ):
    patient_generator = icentia11k.uniform_patient_generator(_str(db_dir), patient_ids, include_labels=False)
    patient_generator = buffered_generator(
        patient_generator, buffer_size=int(num_buffered_patients))
    data_generator = icentia11k.cpc_data_generator(
        patient_generator, context_size=int(context_size), ns=int(ns),
        frame_size=int(frame_size), context_overlap=int(context_overlap),
        positive_offset=int(positive_offset), samples_per_patient=int(samples_per_patient))
    if normalize:
        data_generator = map(lambda x_y: ({
            'context': icentia11k.normalize(x_y[0]['context']),
            'samples': icentia11k.normalize(x_y[0]['samples'])
        }, x_y[1]), data_generator)
    return data_generator


def _str(s):
    return s.decode() if isinstance(s, bytes) else str(s)
