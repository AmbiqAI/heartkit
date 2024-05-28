import logging
import os

import keras
import tensorflow as tf
import tensorflow_model_optimization as tfmot
import wandb
from wandb.keras import WandbMetricsLogger, WandbModelCheckpoint

from ... import tflite as tfa
from ...defines import HKTrainParams
from ...utils import env_flag, set_random_seed, setup_logger
from ..utils import load_datasets
from .datasets import load_train_datasets
from .utils import create_model

logger = setup_logger(__name__)


def train(params: HKTrainParams):
    """Train model

    Args:
        params (HKTrainParams): Training parameters
    """

    params.seed = set_random_seed(params.seed)
    logger.info(f"Random seed {params.seed}")

    os.makedirs(params.job_dir, exist_ok=True)
    logger.info(f"Creating working directory in {params.job_dir}")

    handler = logging.FileHandler(params.job_dir / "train.log", mode="w")
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)

    with open(params.job_dir / "train_config.json", "w", encoding="utf-8") as fp:
        fp.write(params.model_dump_json(indent=2))

    if env_flag("WANDB"):
        wandb.init(
            project=params.project,
            entity="ambiq",
            dir=params.job_dir,
        )
        wandb.config.update(params.model_dump())
    # END IF

    feat_shape = (params.frame_size, 1)
    class_shape = (params.frame_size, 1)

    ds_spec = (
        tf.TensorSpec(shape=feat_shape, dtype=tf.float32),
        tf.TensorSpec(shape=class_shape, dtype=tf.float32),
    )

    datasets = load_datasets(datasets=params.datasets)

    train_ds, val_ds = load_train_datasets(
        datasets=datasets,
        params=params,
        ds_spec=ds_spec,
    )

    with tfa.get_strategy().scope():
        inputs = keras.Input(shape=ds_spec[0].shape, batch_size=None, name="input", dtype=ds_spec[0].dtype)
        if params.resume and params.model_file:
            logger.info(f"Loading model from file {params.model_file}")
            model = tfa.load_model(params.model_file)
            params.model_file = None
        else:
            logger.info("Creating model from scratch")
            model = create_model(
                inputs,
                num_classes=params.num_classes,
                architecture=params.architecture,
            )
        # END IF

        if params.lr_cycles > 1:
            scheduler = keras.optimizers.schedules.CosineDecayRestarts(
                initial_learning_rate=params.lr_rate,
                first_decay_steps=int(0.1 * params.steps_per_epoch * params.epochs),
                t_mul=1.65 / (0.1 * params.lr_cycles * (params.lr_cycles - 1)),
                m_mul=0.4,
            )
        else:
            scheduler = keras.optimizers.schedules.CosineDecay(
                initial_learning_rate=params.lr_rate, decay_steps=params.steps_per_epoch * params.epochs
            )
        # END IF

        optimizer = keras.optimizers.Adam(scheduler)
        loss = keras.losses.MeanSquaredError()

        metrics = [
            keras.metrics.MeanAbsoluteError(name="mae"),
            keras.metrics.MeanSquaredError(name="mse"),
            keras.metrics.CosineSimilarity(name="cosine"),
        ]

        if params.resume and params.weights_file:
            logger.info(f"Hydrating model weights from file {params.weights_file}")
            model.load_weights(params.weights_file)

        if params.model_file is None:
            params.model_file = params.job_dir / "model.keras"

        # Perform QAT if requested
        if params.quantization.enabled and params.quantization.qat:
            logger.info("Performing QAT")
            model = tfmot.quantization.keras.quantize_model(model, quantized_layer_name_prefix="q_")
        # END IF

        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        flops = tfa.get_flops(model, batch_size=1, fpath=params.job_dir / "model_flops.log")
        model(inputs)
        model.summary(print_fn=logger.info)
        logger.info(f"Model requires {flops/1e6:0.2f} MFLOPS")

        ModelCheckpoint = keras.callbacks.ModelCheckpoint
        if env_flag("WANDB"):
            ModelCheckpoint = WandbModelCheckpoint
        model_callbacks = [
            keras.callbacks.EarlyStopping(
                monitor=f"val_{params.val_metric}",
                patience=max(int(0.25 * params.epochs), 1),
                mode="max" if params.val_metric == "f1" else "auto",
                restore_best_weights=True,
            ),
            ModelCheckpoint(
                filepath=str(params.model_file),
                monitor=f"val_{params.val_metric}",
                save_best_only=True,
                save_weights_only=False,
                mode="max" if params.val_metric == "f1" else "auto",
                verbose=1,
            ),
            keras.callbacks.CSVLogger(params.job_dir / "history.csv"),
        ]
        if env_flag("TENSORBOARD"):
            model_callbacks.append(
                keras.callbacks.TensorBoard(
                    log_dir=params.job_dir,
                    write_steps_per_second=True,
                )
            )
        if env_flag("WANDB"):
            model_callbacks.append(WandbMetricsLogger())

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

        logger.info(f"Model saved to {params.model_file}")

        # Get full validation results
        keras.models.load_model(params.model_file)
        logger.info("Performing full validation")
    # END WITH
