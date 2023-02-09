import os

import pydantic_argparse
import tensorflow as tf
import wandb

# import tensorflow_addons as tfa
from keras.engine.keras_tensor import KerasTensor
from wandb.keras import WandbCallback

from neuralspot.tflite.metrics import get_flops

from .datasets.ludb import LudbDataset
from .models.optimizers import Adam
from .models.utils import get_strategy
from .types import HeartTask, HeartTrainParams
from .utils import env_flag, set_random_seed, setup_logger

logger = setup_logger(__name__)


def load_model(inputs: KerasTensor, num_classes: int) -> tf.keras.Model:
    """Load u-net style model.

    Args:
        inputs (KerasTensor): Model input
        num_classes (int): # classes

    Returns:
        tf.keras.Model: Model
    """
    # Entry block
    x = tf.keras.layers.Conv1D(24, 3, strides=2, padding="same")(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    # Blocks 1, 2, 3 are identical apart from the feature depth.
    for filters in [48, 96, 128]:
        x = tf.keras.layers.Activation("relu")(x)
        x = tf.keras.layers.SeparableConv1D(filters, 3, padding="same")(x)
        x = tf.keras.layers.BatchNormalization()(x)

        x = tf.keras.layers.Activation("relu")(x)
        x = tf.keras.layers.SeparableConv1D(filters, 3, padding="same")(x)
        x = tf.keras.layers.BatchNormalization()(x)

        x = tf.keras.layers.MaxPooling1D(3, strides=2, padding="same")(x)

        # Project residual
        residual = tf.keras.layers.Conv1D(filters, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = tf.keras.layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    ### [Second half of the network: upsampling inputs] ###

    for filters in [128, 96, 48, 24]:
        x = tf.keras.layers.Activation("relu")(x)
        x = tf.keras.layers.Conv1DTranspose(filters, 3, padding="same")(x)
        x = tf.keras.layers.BatchNormalization()(x)

        x = tf.keras.layers.Activation("relu")(x)
        x = tf.keras.layers.Conv1DTranspose(filters, 3, padding="same")(x)
        x = tf.keras.layers.BatchNormalization()(x)

        x = tf.keras.layers.UpSampling1D(2)(x)

        # Project residual
        residual = tf.keras.layers.UpSampling1D(2)(previous_block_activation)
        residual = tf.keras.layers.Conv1D(filters, 1, padding="same")(residual)
        x = tf.keras.layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    # Add a per-pixel classification layer
    outputs = tf.keras.layers.Conv1D(num_classes, 3, activation=None, padding="same")(x)

    # Define the model
    model = tf.keras.Model(inputs, outputs)
    return model


def train_model(params: HeartTrainParams):
    """Train model command. This trains a model on the given task and dataset.

    Args:
        params (HeartTrainParams): Training parameters
    """

    params.seed = set_random_seed(params.seed)
    logger.info(f"Random seed {params.seed}")

    strategy = get_strategy()

    os.makedirs(str(params.job_dir), exist_ok=True)
    logger.info(f"Creating working directory in {params.job_dir}")
    with open(str(params.job_dir / "train_config.json"), "w", encoding="utf-8") as fp:
        fp.write(params.json(indent=2))

    if env_flag("WANDB"):
        wandb.init(
            project=f"ecg-{params.task}", entity="ambiq", dir=str(params.job_dir)
        )
        wandb.config.update(params.dict())

    ds = LudbDataset(
        str(params.ds_path), task=params.task, frame_size=params.frame_size
    )
    # Create TF datasets
    with strategy.scope():
        train_ds, val_ds = ds.load_train_datasets(
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

    total_steps = params.steps_per_epoch * params.epochs
    lr_scheduler = tf.keras.optimizers.schedules.CosineDecayRestarts(
        initial_learning_rate=1e-3,
        first_decay_steps=int(0.1 * total_steps),
        t_mul=1.661,  # Creates 4 cycles
        m_mul=0.50,
    )

    with strategy.scope():
        logger.info("Building model")
        inputs = tf.keras.Input(
            shape=(params.frame_size, 1), batch_size=None, dtype=tf.float32
        )
        model = load_model(inputs=inputs, num_classes=4)
        # create_task_model(
        #     inputs, params.task, params.arch, stages=params.stages
        # )
        flops = get_flops(model, batch_size=1)
        optimizer = Adam(lr_scheduler)
        model.compile(
            # optimizer="rmsprop",
            optimizer=optimizer,
            # loss=tfa.losses.SigmoidFocalCrossEntropy(from_logits=True),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="acc")],
        )
        model(inputs)
        model.summary()
        logger.info(f"Model requires {flops/1e6:0.2f} MFLOPS")

        if params.weights_file:
            logger.info(f"Loading weights from file {params.weights_file}")
            model.load_weights(str(params.weights_file))
        params.weights_file = str(params.job_dir / "model.weights")

        model_callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor=f"val_{params.val_metric}",
                patience=40,
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
    # END WITH


def create_parser():
    """Create CLI argument parser
    Returns:
        ArgumentParser: Arg parser
    """
    return pydantic_argparse.ArgumentParser(
        model=HeartTrainParams,
        prog="Heart Train Command",
        description="Train heart model",
    )


if __name__ == "__main__":
    # parser = create_parser()
    # train_model(parser.parse_typed_args())
    # with tf.device('/cpu:0'):
    train_model(
        HeartTrainParams(
            task=HeartTask.segmentation,
            job_dir="./results/segmentation",
            ds_path="./datasets",
            frame_size=1248,
            samples_per_patient=800,
            val_samples_per_patient=800,
            val_patients=0.10,
            val_size=8_000,
            batch_size=64,
            buffer_size=8092,
            epochs=100,
            steps_per_epoch=600,  # (200*0.9*400)/64
            val_metric="loss",
            data_parallelism=8,
        )
    )
