import logging
import os

import keras
import tensorflow as tf
import wandb
from wandb.keras import WandbMetricsLogger, WandbModelCheckpoint

import neuralspot_edge as nse
from ...defines import HKTrainParams
from ...models import ModelFactory
from ...utils import env_flag, set_random_seed, setup_logger
from ..utils import load_datasets
from .datasets import load_train_datasets

logger = setup_logger(__name__)


def train(params: HKTrainParams):
    """Train  model

    Args:
        params (HKTrainParams): Training parameters
    """

    params.temperature = float(getattr(params, "temperature", 0.1))

    params.seed = set_random_seed(params.seed)
    logger.debug(f"Random seed {params.seed}")

    os.makedirs(params.job_dir, exist_ok=True)
    logger.debug(f"Creating working directory in {params.job_dir}")

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

    # Currently we return positive pairs w/o labels
    feat_shape = (params.frame_size, 1)
    ds_spec = (
        tf.TensorSpec(shape=feat_shape, dtype="float32"),
        tf.TensorSpec(shape=feat_shape, dtype="float32"),
    )

    datasets = load_datasets(datasets=params.datasets)

    train_ds, val_ds = load_train_datasets(
        datasets=datasets,
        params=params,
        ds_spec=ds_spec,
    )

    projection_width = params.num_classes

    encoder_input = keras.Input(shape=feat_shape, dtype="float32")

    # Encoder
    encoder = ModelFactory.get(params.architecture.name)(
        x=encoder_input,
        params=params.architecture.params,
        num_classes=None,
    )

    encoder_output = encoder(encoder_input)
    flops = nse.metrics.flops.get_flops(encoder, batch_size=1, fpath=params.job_dir / "encoder_flops.log")
    encoder.summary(print_fn=logger.info)
    logger.debug(f"Encoder requires {flops/1e6:0.2f} MFLOPS")

    # Projector
    projector_input = encoder_output
    projector_output = keras.layers.Dense(projection_width, activation="relu6")(projector_input)
    projector_output = keras.layers.Dense(projection_width)(projector_output)
    projector = keras.Model(inputs=projector_input, outputs=projector_output, name="projector")
    flops = nse.metrics.flops.get_flops(projector, batch_size=1, fpath=params.job_dir / "projector_flops.log")
    projector.summary(print_fn=logger.info)
    logger.debug(f"Projector requires {flops/1e6:0.2f} MFLOPS")

    if params.model_file is None:
        params.model_file = params.job_dir / "model.keras"

    model = nse.models.opimizers.simclr.SimCLR(
        contrastive_augmenter=lambda x: x,
        encoder=encoder,
        projector=projector,
        # momentum_coeff=0.999,
        temperature=params.temperature,
        # queue_size=65536,
    )

    def get_scheduler():
        if params.lr_cycles > 1:
            return keras.optimizers.schedules.CosineDecayRestarts(
                initial_learning_rate=params.lr_rate,
                first_decay_steps=int(0.1 * params.steps_per_epoch * params.epochs),
                t_mul=1.65 / (0.1 * params.lr_cycles * (params.lr_cycles - 1)),
                m_mul=0.4,
            )
        return keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=params.lr_rate,
            decay_steps=params.steps_per_epoch * params.epochs,
        )

    model.compile(
        contrastive_optimizer=keras.optimizers.Adam(get_scheduler()),
        probe_optimizer=keras.optimizers.Adam(get_scheduler()),
    )

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

    logger.debug(f"Model saved to {params.model_file}")
