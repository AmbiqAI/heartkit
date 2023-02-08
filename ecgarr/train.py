import os

import numpy as np
import pydantic_argparse
import tensorflow as tf
import tensorflow_model_optimization as tfmot
import wandb
from sklearn.metrics import f1_score
from wandb.keras import WandbCallback

from neuralspot.tflite.metrics import get_flops

from . import datasets as ds
from .metrics import confusion_matrix_plot
from .models.optimizers import Adam
from .models.utils import generate_task_model, get_strategy
from .types import EcgTrainParams
from .utils import env_flag, set_random_seed, setup_logger

logger = setup_logger(__name__)


def train_model(params: EcgTrainParams):
    """Train model command. This trains a model on the given task and dataset.

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
        wandb.init(
            project=f"ecg-{params.task}", entity="ambiq", dir=str(params.job_dir)
        )
        wandb.config.update(params.dict())

    # Create TF datasets
    train_ds, val_ds = ds.load_datasets(
        ds_path=str(params.ds_path),
        task=params.task,
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
    train_ds = (
        train_ds.shuffle(
            buffer_size=params.buffer_size,
            reshuffle_each_iteration=True,
        )
        .batch(
            batch_size=params.batch_size,
            drop_remainder=True,
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
        )
        .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    )
    val_ds = val_ds.batch(
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

    strategy = get_strategy()
    with strategy.scope():
        logger.info("Building model")
        inputs = tf.keras.Input(
            shape=(params.frame_size, 1), batch_size=None, dtype=tf.float32
        )
        model = generate_task_model(
            inputs, params.task, params.arch, stages=params.stages
        )
        flops = get_flops(model, batch_size=1)
        model.compile(
            optimizer=Adam(learning_rate=5e-4, beta_1=0.9, beta_2=0.98, epsilon=1e-9),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="acc")],
        )
        model(inputs)
        # logger.info(f"# model parameters: {model.count_params()}")
        model.summary()
        logger.info(f"Model requires {flops/1e6:0.2f} MFLOPS")

        if params.weights_file:
            logger.info(f"Loading weights from file {params.weights_file}")
            model.load_weights(str(params.weights_file))
        params.weights_file = str(params.job_dir / "model.weights")

        model_callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor=f"val_{params.val_metric}",
                patience=10,
                mode="max" if params.val_metric == "f1" else "auto",
                restore_best_weights=True,
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=params.weights_file,
                monitor=f"val_{params.val_metric}",
                save_best_only=True,
                save_weights_only=True,
                mode="max" if params.val_metric == "f1" else "auto",
                verbose=1,
            ),
            tf.keras.callbacks.CSVLogger(str(params.job_dir / "history.csv")),
            tf.keras.callbacks.TensorBoard(
                log_dir=str(params.job_dir), write_steps_per_second=True
            ),
            tf.keras.callbacks.LearningRateScheduler(decay),
        ]
        if env_flag("WANDB"):
            model_callbacks.append(WandbCallback())

        if params.epochs:
            try:
                model.fit(
                    train_ds,
                    steps_per_epoch=params.steps_per_epoch,
                    verbose=2,
                    epochs=params.epochs,
                    validation_data=val_ds,
                    callbacks=model_callbacks,
                )
            except KeyboardInterrupt:
                logger.warning("Stopping training due to keyboard interrupt")

            # Restore best weights from checkpoint
            model.load_weights(params.weights_file)

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
                    train_ds,
                    steps_per_epoch=params.steps_per_epoch,
                    verbose=2,
                    epochs=max(1, params.epochs // 10),
                    validation_data=val_ds,
                    callbacks=model_callbacks,
                )
            model = q_model

        # Get full validation results
        logger.info("Performing full validation")
        test_labels = []
        for _, label in val_ds:
            test_labels.append(label.numpy())
        y_true = np.concatenate(test_labels)
        y_pred = np.argmax(model.predict(val_ds), axis=1)

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
        prog="Heart Train Command",
        description="Train heart model",
    )


if __name__ == "__main__":
    parser = create_parser()
    train_model(parser.parse_typed_args())
