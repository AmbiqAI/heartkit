import logging
from typing import Optional
import numpy as np
import tensorflow as tf
import pydantic_argparse
from . import datasets as ds
from .metrics import confusion_matrix_plot
from .utils import setup_logger
from .types import EcgTask, EcgTestParams

logger = logging.getLogger('ecgarr.test')

@tf.function
def parallelize_dataset(
        db_path: str,
        patient_ids: int = None,
        task: EcgTask = EcgTask.rhythm,
        frame_size: int = 1250,
        samples_per_patient: int = 100,
        repeat: bool = False,
        num_workers: int = 1
):
    def _make_train_dataset(i, split):
        return ds.create_dataset_from_generator(
            task=task, db_path=db_path,
            patient_ids=patient_ids[i * split:(i + 1) * split], frame_size=frame_size,
            samples_per_patient=samples_per_patient, repeat=repeat
        )
    split = len(patient_ids) // num_workers
    datasets = [_make_train_dataset(i, split) for i in range(num_workers)]
    par_ds = tf.data.Dataset.from_tensor_slices(datasets)
    return par_ds.interleave(lambda x: x, cycle_length=num_workers, num_parallel_calls=tf.data.experimental.AUTOTUNE)


def load_test_dataset(
        db_path: str,
        task: str = 'rhythm',
        frame_size: int = 1250,
        test_patients: Optional[float] = None,
        test_pt_samples: Optional[int] = None,
        num_workers: int = 1
    ):
    test_patient_ids = ds.icentia11k.get_test_patient_ids()
    if test_patients is not None:
        num_pts = int(test_patients) if test_patients > 1 else int(test_patients*len(test_patient_ids))
        test_patient_ids = test_patient_ids[:num_pts]
    # np.random.shuffle(test_patient_ids)

    test_size = len(test_patient_ids) * test_pt_samples * 4
    logger.info(f'Collecting {test_size} test samples')
    test_patient_ids = tf.convert_to_tensor(test_patient_ids)
    test_data = parallelize_dataset(
        db_path=db_path, patient_ids=test_patient_ids, task=task, frame_size=frame_size,
        samples_per_patient=test_pt_samples, repeat=False, num_workers=num_workers
    )
    test_x, test_y = next(test_data.batch(test_size).as_numpy_iterator())
    test_data = ds.create_dataset_from_data(test_x, test_y, task=task, frame_size=frame_size)
    return test_data

def evaluate_model(params: EcgTestParams):
    setup_logger('ecgarr', str(params.job_dir))
    test_data = load_test_dataset(
        db_path=str(params.db_path),
        task=params.task,
        frame_size=params.frame_size,
        test_patients=params.test_patients,
        test_pt_samples=params.samples_per_patient,
        num_workers=params.data_parallelism
    )
    test_data = test_data.batch(
        batch_size=512,
        drop_remainder=True,
        num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        model = tf.keras.models.load_model(params.model_file)
        model.summary()
        test_labels = []
        for _, label in test_data:
            test_labels.append(label.numpy())
        y_true = np.concatenate(test_labels)
        y_pred = np.argmax(model.predict(test_data), axis=1)

        # Summarize results
        class_names = ['norm', 'afib']
        test_acc = np.sum(y_pred == y_true) / len(y_true)
        logger.info(f'Test set accuracy: {test_acc:.0%}')
        # TODO: Report accuracy, f1, precision, sensitivity, specificity
        confusion_matrix_plot(y_true, y_pred, labels=class_names, save_path=str(params.job_dir / 'confusion_matrix_test.png'))

def create_parser():
    return pydantic_argparse.ArgumentParser(
        model=EcgTestParams,
        prog="ECG Arrhythmia Testing Params",
        description="ECG Arrhythmia Testing Params"
    )

if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_typed_args()
    evaluate_model(args)
