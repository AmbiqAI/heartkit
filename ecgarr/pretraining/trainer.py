import os
import sys
import logging
import random
import argparse
from pathlib import Path
from typing import Any, List, Optional, Tuple
import numpy as np
import sklearn.model_selection
import wandb
from wandb.keras import WandbCallback
import tensorflow as tf
import tensorflow_model_optimization as tfmot
import matplotlib.pyplot as plt
import numpy.typing as npt
import seaborn as sns
from . import datasets
from .utils import task_solver
from ..datasets import icentia11k
from ..evaluation import CustomCheckpoint, f1
from ..models.utils import build_input_tensor_from_shape
from ..utils import EcgTask, matches_spec, load_pkl, save_pkl, xxd_c_dump

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger('ecgarr.train')

def confusion_matrix_plot(y_true: npt.ArrayLike, y_pred: npt.ArrayLike, labels: List[str], save_path: Optional[str], **kwargs):
        confusion_mtx = tf.math.confusion_matrix(y_true, y_pred)
        plt.figure(figsize=kwargs.get('figsize', (10, 8)))
        sns.heatmap(confusion_mtx, xticklabels=labels, yticklabels=labels, annot=True, fmt='g')
        plt.xlabel('Prediction')
        plt.ylabel('Label')
        if save_path:
            plt.savefig(save_path)
        plt.close()

def _split_train_test_patients(task: EcgTask, patient_ids: npt.ArrayLike, test_size: float) -> List[List[Any]]:
    if task == EcgTask.rhythm:
        return icentia11k.train_test_split_patients(patient_ids, test_size=test_size, task=task)
    return sklearn.model_selection.train_test_split(patient_ids, test_size=test_size)

def _create_dataset_from_generator(task: EcgTask, db_path: str, patient_ids: List[int], frame_size: int, samples_per_patient: int = 1, repeat: bool = True):
    if task == EcgTask.rhythm:
        dataset = datasets.rhythm_dataset(
            db_path=db_path, patient_ids=patient_ids, frame_size=frame_size,
            samples_per_patient=samples_per_patient, repeat=repeat
        )
    elif task == EcgTask.beat:
        dataset = datasets.beat_dataset(
            db_path=db_path, patient_ids=patient_ids, frame_size=frame_size,
            samples_per_patient=samples_per_patient
        )
    elif task == EcgTask.hr:
        dataset = datasets.heart_rate_dataset(
            db_path=db_path, patient_ids=patient_ids, frame_size=frame_size,
            samples_per_patient=samples_per_patient
        )
    else:
        raise ValueError('unknown task: {}'.format(task))
    return dataset

def _create_dataset_from_data(x: npt.ArrayLike, y: npt.ArrayLike, task: EcgTask, frame_size: int):
    if task in [EcgTask.rhythm, EcgTask.beat, EcgTask.hr]:
        spec = (tf.TensorSpec((None, frame_size, 1), tf.float32), tf.TensorSpec((None,), tf.int32))
    else:
        raise ValueError('unknown task: {}'.format(task))
    if not matches_spec((x, y), spec, ignore_batch_dim=True):
        raise ValueError('data does not match the required spec: {}'.format(spec))
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    return dataset

def validate_args(args):
    if args.val_metric not in ['loss', 'acc', 'f1']:
        raise ValueError('Unknown metric: {}'.format(args.val_metric))

def initialize_seed(seed: Optional[int]):
    seed = seed or np.random.randint(2 ** 16)
    logger.info(f'Setting random state {seed}')
    np.random.seed(seed)
    random.seed(seed)

def load_datasets(
        db_path: str,
        task: str = 'rhythm',
        frame_size: int = 1250,
        train_patients: Optional[float] = None,
        val_patients: Optional[float] = None,
        train_pt_samples: Optional[int] = None,
        val_pt_samples: Optional[int] = None,
        train_file: Optional[str] = None,
        val_file: Optional[str] = None,
        num_workers: int = 1
    ) -> Tuple[tf.data.Dataset, tf.data.Dataset]:

    if val_patients is not None and val_patients >= 1:
        val_patients = int(val_patients)

    train_pt_samples = train_pt_samples or 1000
    if val_pt_samples is None:
        val_pt_samples = train_pt_samples

    # Get train patients
    train_patient_ids = icentia11k.ds_patient_ids
    if train_patients is not None:
        num_pts = int(train_patients) if train_patients > 1 else int(train_patients*len(train_patient_ids))
        train_patient_ids = train_patient_ids[:num_pts]
    np.random.shuffle(train_patient_ids)

    # if train_file and os.path.isfile(train_file):
    #     logger.info(f'Loading training data from file {train_file}')
    #     train = load_pkl(train_file)
    #     train_data = _create_dataset_from_data(train['x'], train['y'], task=task, frame_size=frame_size)
    #     train_patient_ids = train['patient_ids']

    if val_file and os.path.isfile(val_file):
        logger.info(f'Loading validation data from file {val_file}')
        val = load_pkl(val_file)
        validation_data = _create_dataset_from_data(val['x'], val['y'], task=task, frame_size=frame_size)
        val_patient_ids = val['patient_ids']
        # remove patients who belong to the validation set from train data
        train_patient_ids = np.setdiff1d(train_patient_ids, val_patient_ids)
    else:
        logger.info('Splitting patients into train and validation')
        train_patient_ids, val_patient_ids = _split_train_test_patients(
            task=task, patient_ids=train_patient_ids, test_size=val_patients
        )

        # validation size is one validation epoch by default
        val_size = len(val_patient_ids) * val_pt_samples
        logger.info(f'Collecting {val_size} validation samples')
        split = len(val_patient_ids) // num_workers
        val_patient_ids = tf.convert_to_tensor(val_patient_ids)
        validation_data = tf.data.Dataset.range(num_workers).interleave(lambda i: _create_dataset_from_generator(
            task=task, db_path=db_path,
            patient_ids=val_patient_ids[i * split:(i + 1) * split], frame_size=frame_size,
            samples_per_patient=val_pt_samples, repeat=False
        ), num_parallel_calls=tf.data.experimental.AUTOTUNE)
        val_x, val_y = next(validation_data.batch(val_size).as_numpy_iterator())
        validation_data = _create_dataset_from_data(val_x, val_y, task=task, frame_size=frame_size)
        val = dict(x=val_x, y=val_y, patient_ids=val_patient_ids)

        # Cache validation set
        if val_file:
            os.makedirs(os.path.dirname(val_file), exist_ok=True)
            logger.info(f'Caching the validation set in {val_file}')
            save_pkl(val_file, x=val_x, y=val_y, patient_ids=val_patient_ids)
        # END IF
    # END IF

    logger.info('Building train data generators')
    split = len(train_patient_ids) // num_workers
    train_patient_ids = tf.convert_to_tensor(train_patient_ids)
    train_data = tf.data.Dataset.range(num_workers).interleave(
        lambda i: _create_dataset_from_generator(
            task=task, db_path=db_path,
            patient_ids=train_patient_ids[i * split:(i + 1) * split], frame_size=frame_size,
            samples_per_patient=train_pt_samples, repeat=True
        ),
        num_parallel_calls=tf.data.experimental.AUTOTUNE)

    return train_data, validation_data, val

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Task arguments
    parser.add_argument('--task', required=True, help='Training task: `rhythm`, `beat`, `hr`.', default='rhythm')
    parser.add_argument('--job-dir', type=Path, required=True, help='Job output directory.')
    # Dataset arguments
    parser.add_argument('--db-path', type=Path, required=True, help='Path to database directory')
    parser.add_argument('--frame-size', type=int, default=2048, help='ECG frame size')
    parser.add_argument('--samples-per-patient', type=int, default=1000, help='# train samples per patient')
    parser.add_argument('--val-samples-per-patient', type=int, default=None, help='# validation samples per patient')
    parser.add_argument('--train-patients', type=float, default=None, help='# or proportion of patients for training')
    parser.add_argument('--val-patients', type=float, default=None, help='# or proportion of patients for validation')
    parser.add_argument('--val-file', type=Path, help='Path to load/store pickled validation file')
    parser.add_argument('--data-parallelism', type=int, default=1, help='# of data loaders running in parallel')
    # Model arguments
    parser.add_argument('--weights-file', type=Path, help='Path to a checkpoint weights to load')
    parser.add_argument('--arch', default='resnet12', help='Network architecture: `resnet12`, `resnet18`, `resnet34` or `resnet50`')
    parser.add_argument('--stages', type=int, default=None, help='# of resnet stages')
    parser.add_argument('--quantization', action=argparse.BooleanOptionalAction, help='Enable post quantization')
    # Training arguments
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size.')
    parser.add_argument('--buffer-size', type=int, default=100, help='Buffer size.')
    parser.add_argument('--epochs', type=int, default=1, help='Number of epochs.')
    parser.add_argument('--steps-per-epoch', type=int, default=100, help='Number of steps per epoch.')
    parser.add_argument('--val-metric', default='loss', help='Performance metric: `loss`, `acc` or `f1`')
    # Extra arguments
    parser.add_argument('--seed', type=int, default=None, help='Random state seed')
    parser.add_argument('--wandb', action=argparse.BooleanOptionalAction, help='Enable wandb callback')

    args, _ = parser.parse_known_args()
    validate_args(args=args)

    initialize_seed(args.seed)

    os.makedirs(str(args.job_dir), exist_ok=True)
    logger.info(f'Creating working directory in {args.job_dir}')

    if args.wandb:
        wandb.init(project="ecg-arrhythmia", entity="ambiq", dir=str(args.job_dir / 'wandb'), )
        wandb.config.update(args)

    # Load datasets
    train_data, validation_data, val = load_datasets(
        db_path=str(args.db_path),
        frame_size=args.frame_size,
        train_patients=args.train_patients,
        val_patients=args.val_patients,
        train_pt_samples=args.samples_per_patient,
        val_pt_samples=args.val_samples_per_patient,
        val_file=args.val_file,
        num_workers=args.data_parallelism
    )
    # Shuffle and batch datasets for training
    train_data = train_data.prefetch(tf.data.experimental.AUTOTUNE).shuffle(args.buffer_size or 100)
    train_data = train_data.batch(args.batch_size)
    validation_data = validation_data.batch(args.batch_size)

    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        logger.info('Building model')
        model = task_solver(args.task, args.arch, stages=args.stages)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(beta_1=0.9, beta_2=0.98, epsilon=1e-9),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name='acc')]
        )

        input_shape, _ = tf.compat.v1.data.get_output_shapes(train_data)
        input_dtype, _ = tf.compat.v1.data.get_output_types(train_data)
        inputs = build_input_tensor_from_shape(input_shape, dtype=input_dtype, ignore_batch_dim=True)
        model(inputs)

        logger.info(f'# model parameters: {model.count_params()}')
        model.summary()

        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', min_delta=0, patience=10, verbose=0,
            mode='auto', restore_best_weights=True
        )

        if args.weights_file:
            logger.info(f'Loading weights from file {args.weights_file}')
            model.load_weights(str(args.weights_file))

        if args.val_metric in ['loss', 'acc']:
            checkpoint = tf.keras.callbacks.ModelCheckpoint(
                filepath=str(args.job_dir / 'epoch_{epoch:02d}' / 'model.weights'),
                monitor=f'val_{args.val_metric}',
                save_best_only=False, save_weights_only=True,
                mode='max' if args.val_metric == 'f1' else 'auto', verbose=1
            )
        elif args.val_metric == 'f1':
            checkpoint = CustomCheckpoint(
                filepath=str(args.job_dir / 'epoch_{epoch:02d}' / 'model.weights'),
                data=(validation_data, val['y']),
                score_fn=f1, save_best_only=False, verbose=1
            )
        else:
            raise ValueError('Unknown metric: {}'.format(args.val_metric))
        tf_logger = tf.keras.callbacks.CSVLogger(str(args.job_dir / 'history.csv'))
        model_callbacks = [early_stopping, checkpoint, tf_logger]
        if args.wandb:
            model_callbacks.append(WandbCallback())

        if args.epochs:
            model.fit(
                train_data, steps_per_epoch=args.steps_per_epoch, verbose=2, epochs=args.epochs,
                validation_data=validation_data, callbacks=model_callbacks
            )

        if args.quantization:
            quantize_model = tfmot.quantization.keras.quantize_model
            q_model = quantize_model(model)
            q_model.compile(
                optimizer=tf.keras.optimizers.Adam(beta_1=0.9, beta_2=0.98, epsilon=1e-9),
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name='acc')]
            )
            logger.info(f'# model parameters: {model.count_params()}')
            q_model.summary()
            if args.epochs:
                q_model.fit(
                    train_data, steps_per_epoch=args.steps_per_epoch, verbose=2, epochs=args.epochs,
                    validation_data=validation_data, callbacks=model_callbacks
                )

        tf_model_path = str(args.job_dir / 'model.tf')
        tfl_model_path = str(args.job_dir / 'model.tflite')
        tflm_model_path = str(args.job_dir / 'model.cc')

        # Save model
        model.save(tf_model_path)

        # TFLite converter
        converter = tf.lite.TFLiteConverter.from_saved_model(tf_model_path)
        # Perform quantization
        if args.quantization:
            def representative_dataset():
                for x in list(validation_data.take(1))[0][0][:args.batch_size]:
                    yield([tf.expand_dims(x, axis=0)])
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = tf.uint8
            converter.inference_output_type = tf.uint8
            converter.representative_dataset = representative_dataset
        model_tflite = converter.convert()
        # Save TFLite model
        with open(tfl_model_path, 'wb') as fp:
            fp.write(model_tflite)
        # Save TF Micro model (Generate C array similar to `xxd -i`)
        xxd_c_dump(tfl_model_path, tflm_model_path, var_name='g_model', chunk_len=12)

        test_labels = []
        for _, label in validation_data:
            test_labels.append(label.numpy())
        y_true = np.concatenate(test_labels)
        y_pred = np.argmax(model.predict(validation_data), axis=1)

        # Summarize results
        class_names = ['norm', 'afib']
        test_acc = np.sum(y_pred == y_true) / len(y_true)
        logger.info(f'Test set accuracy: {test_acc:.0%}')
        if args.wandb:
            confusion_matrix_plot(y_true, y_pred, labels=class_names, save_path=str(args.job_dir / 'confusion_matrix.png'))
            wandb.log({"afib_conf_mat" : wandb.plot.confusion_matrix(
                probs=None, preds=y_pred, y_true=y_true, class_names=class_names
            )})
        # END IF
    # END WITH
