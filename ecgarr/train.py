import os
import sys
from typing import List, Optional, Tuple, Union

import numpy as np
import pydantic_argparse
import tensorflow as tf
import tensorflow_model_optimization as tfmot
import wandb
from sklearn.metrics import f1_score
from wandb.keras import WandbCallback

from . import datasets as ds
from .datasets import icentia11k
from .metrics import confusion_matrix_plot
from .models.utils import generate_task_model
from .types import EcgTask, EcgTrainParams
from .utils import env_flag, load_pkl, save_pkl, set_random_seed, setup_logger

logger = setup_logger(__name__)

if sys.platform == "darwin":
    Adam = tf.keras.optimizers.legacy.Adam
else:
    Adam = tf.keras.optimizers.Adam


@tf.function
def parallelize_dataset(
    db_path: str,
    patient_ids: int = None,
    task: EcgTask = EcgTask.rhythm,
    frame_size: int = 1250,
    samples_per_patient: Union[int, List[int]] = 100,
    repeat: bool = False,
    num_workers: int = 1,
):
    """Generates datasets for given task in parallel using TF `interleave`

    Args:
        db_path (str): Database path
        patient_ids (int, optional): List of patient IDs. Defaults to None.
        task (EcgTask, optional): ECG Task routine. Defaults to EcgTask.rhythm.
        frame_size (int, optional): Frame size. Defaults to 1250.
        samples_per_patient (int, optional): # Samples per pateint. Defaults to 100.
        repeat (bool, optional): Should data generator repeat. Defaults to False.
        num_workers (int, optional): Number of parallel workers. Defaults to 1.
    """

    def _make_train_dataset(i, split):
        return ds.create_dataset_from_generator(
            task=task,
            db_path=db_path,
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
    db_path: str,
    task: EcgTask = EcgTask.rhythm,
    frame_size: int = 1250,
    train_patients: Optional[float] = None,
    val_patients: Optional[float] = None,
    train_pt_samples: Optional[Union[int, List[int]]] = None,
    val_pt_samples: Optional[Union[int, List[int]]] = None,
    val_size: Optional[int] = None,
    val_file: Optional[str] = None,
    num_workers: int = 1,
) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    """Load training and validation datasets
    Args:
        db_path (str): Database path
        task (EcgTask, optional): Heart arrhythmia task. Defaults to EcgTask.rhythm.
        frame_size (int, optional): Frame size. Defaults to 1250.
        train_patients (Optional[float], optional): # or proportion of train patients. Defaults to None.
        val_patients (Optional[float], optional): # or proportion of train patients. Defaults to None.
        train_pt_samples (Optional[Union[int, List[int]]], optional): # samples per patient for training. Defaults to None.
        val_pt_samples (Optional[Union[int, List[int]]], optional): # samples per patient for training. Defaults to None.
        train_file (Optional[str], optional): Path to existing picked training file. Defaults to None.
        val_file (Optional[str], optional): Path to existing picked validation file. Defaults to None.
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
        validation_data = ds.create_dataset_from_data(
            val["x"], val["y"], task=task, frame_size=frame_size
        )
        val_patient_ids = val["patient_ids"]
        train_patient_ids = np.setdiff1d(train_patient_ids, val_patient_ids)
    else:
        logger.info("Splitting patients into train and validation")
        train_patient_ids, val_patient_ids = ds.split_train_test_patients(
            task=task, patient_ids=train_patient_ids, test_size=val_patients
        )
        if val_size is None:
            val_size = 250 * len(val_patient_ids)

        logger.info(f"Collecting {val_size} validation samples")
        validation_data = parallelize_dataset(
            db_path=db_path,
            patient_ids=val_patient_ids,
            task=task,
            frame_size=frame_size,
            samples_per_patient=val_pt_samples,
            repeat=False,
            num_workers=num_workers,
        )
        val_x, val_y = next(validation_data.batch(val_size).as_numpy_iterator())
        validation_data = ds.create_dataset_from_data(
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
    train_data = parallelize_dataset(
        db_path=db_path,
        patient_ids=train_patient_ids,
        task=task,
        frame_size=frame_size,
        samples_per_patient=train_pt_samples,
        repeat=True,
        num_workers=num_workers,
    )
    return train_data, validation_data


def train_model(params: EcgTrainParams):
    """Train model command. This trains a ResNet still network on the given task and dataset.

    Args:
        params (EcgTrainParams): Training parameters
    """

    params.seed = set_random_seed(params.seed)
    logger.info(f"Random seed {params.seed}")

    os.makedirs(str(params.job_dir), exist_ok=True)
    logger.info(f"Creating working directory in {params.job_dir}")
    with open(str(params.job_dir / "train_config.json"), "w", encoding="utf-8") as fp:
        fp.write(params.json(indent=2))

    if env_flag("WANDB"):
        wandb.init(project="ecg-arrhythmia", entity="ambiq", dir=str(params.job_dir))
        wandb.config.update(params.dict())

    # Create TF datasets
    train_data, validation_data = load_datasets(
        db_path=str(params.db_path),
        frame_size=params.frame_size,
        train_patients=params.train_patients,
        val_patients=params.val_patients,
        train_pt_samples=params.samples_per_patient,
        val_pt_samples=params.val_samples_per_patient,
        val_file=params.val_file,
        val_size=params.val_size,
        num_workers=params.data_parallelism,
    )
    # Shuffle and batch datasets for training
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = (
        tf.data.experimental.AutoShardPolicy.DATA
    )
    train_data = (
        train_data.shuffle(
            buffer_size=params.buffer_size,
            reshuffle_each_iteration=True,
        )
        .batch(
            batch_size=params.batch_size,
            drop_remainder=True,
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
        )
        .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        .with_options(options)
    )
    validation_data = validation_data.batch(
        batch_size=params.batch_size,
        drop_remainder=True,
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )

    def decay(epoch):
        if epoch < 15:
            return 1e-3
        if epoch < 30:
            return 1e-4
        return 1e-5

    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        logger.info("Building model")
        # NOTE: Leave batch as None so later TFL conversion can set to 1 for inference
        inputs = tf.keras.Input(
            shape=(params.frame_size, 1), batch_size=None, dtype=tf.float32
        )
        model = generate_task_model(
            inputs, params.task, params.arch, stages=params.stages
        )
        model.compile(
            optimizer=Adam(learning_rate=5e-4, beta_1=0.9, beta_2=0.98, epsilon=1e-9),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="acc")],
        )
        model(inputs)

        logger.info(f"# model parameters: {model.count_params()}")
        model.summary()

        if params.weights_file:
            logger.info(f"Loading weights from file {params.weights_file}")
            model.load_weights(str(params.weights_file))

        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor=f"val_{params.val_metric}",
            min_delta=0,
            patience=10,
            verbose=0,
            mode="max" if params.val_metric == "f1" else "auto",
            restore_best_weights=True,
        )

        checkpoint_weight_path = str(params.job_dir / "model.weights")
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_weight_path,
            monitor=f"val_{params.val_metric}",
            save_best_only=True,
            save_weights_only=True,
            mode="max" if params.val_metric == "f1" else "auto",
            verbose=1,
        )
        tf_logger = tf.keras.callbacks.CSVLogger(str(params.job_dir / "history.csv"))
        lr_scheduler = tf.keras.callbacks.LearningRateScheduler(decay)
        model_callbacks = [early_stopping, checkpoint, tf_logger, lr_scheduler]
        if env_flag("WANDB"):
            model_callbacks.append(WandbCallback())

        if params.epochs:
            try:
                model.fit(
                    train_data,
                    steps_per_epoch=params.steps_per_epoch,
                    verbose=2,
                    epochs=params.epochs,
                    validation_data=validation_data,
                    callbacks=model_callbacks,
                )
            except KeyboardInterrupt:
                logger.warning("Stopping training due to keyboard interrupt")

            # Restore best weights from checkpoint
            model.load_weights(checkpoint_weight_path)

        # Save full model
        tf_model_path = str(params.job_dir / "model.tf")
        logger.info(f"Model saved to {tf_model_path}")
        model.save(tf_model_path)

        # Perform QAT fine-tuning
        if params.quantization:
            q_model = tfmot.quantization.keras.quantize_model(model)
            q_model.compile(
                optimizer=tf.keras.optimizers.Adam(
                    learning_rate=5e-6, beta_1=0.9, beta_2=0.98, epsilon=1e-9
                ),
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="acc")],
            )
            logger.info(f"# qmodel parameters: {q_model.count_params()}")
            q_model.summary()
            if params.epochs:
                q_model.fit(
                    train_data,
                    steps_per_epoch=params.steps_per_epoch,
                    verbose=2,
                    epochs=max(1, params.epochs // 10),
                    validation_data=validation_data,
                    callbacks=model_callbacks,
                )
            model = q_model

        # Get full validation results
        logger.info("Performing full validation")
        test_labels = []
        for _, label in validation_data:
            test_labels.append(label.numpy())
        y_true = np.concatenate(test_labels)
        y_pred = np.argmax(model.predict(validation_data), axis=1)

        # Summarize results
        class_names = ds.get_class_names(task=params.task)
        test_acc = np.sum(y_pred == y_true) / len(y_true)
        test_f1 = f1_score(y_true, y_pred, average="macro")
        logger.info(f"[VAL SET] ACC={test_acc:.2%}, F1={test_f1:.2%}")
        confusion_matrix_plot(
            y_true,
            y_pred,
            labels=class_names,
            save_path=str(params.job_dir / "confusion_matrix.png"),
        )
        if env_flag("WANDB"):
            wandb.log(
                {
                    "afib_conf_mat": wandb.plot.confusion_matrix(
                        probs=None, preds=y_pred, y_true=y_true, class_names=class_names
                    )
                }
            )
        # END IF
    # END WITH


def create_parser():
    """Create CLI argument parser
    Returns:
        ArgumentParser: Arg parser
    """
    return pydantic_argparse.ArgumentParser(
        model=EcgTrainParams,
        prog="Heart Arrhythmia Train Command",
        description="Train heart arrhythmia model",
    )


if __name__ == "__main__":
    parser = create_parser()
    train_model(parser.parse_typed_args())
