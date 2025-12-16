import os

import keras
import helia_edge as helia
import numpy as np
import wandb
from wandb.integration.keras import WandbMetricsLogger, WandbModelCheckpoint

from ...defines import HKTaskParams
from ...datasets import DatasetFactory
from ...models import ModelFactory
from .datasets import load_train_datasets


def train(params: HKTaskParams):
    """Train translate task model with given parameters.

    Args:
        params (HKTaskParams): Task parameters
    """
    os.makedirs(params.job_dir, exist_ok=True)
    logger = helia.utils.setup_logger(__name__, level=params.verbose, file_path=params.job_dir / "train.log")
    logger.debug(f"Creating working directory in {params.job_dir}")

    params.seed = helia.utils.set_random_seed(params.seed)
    logger.debug(f"Random seed {params.seed}")

    with open(params.job_dir / "configuration.json", "w", encoding="utf-8") as fp:
        fp.write(params.model_dump_json(indent=2))

    if helia.utils.env_flag("WANDB"):
        wandb.init(
            project=params.project,
            entity="ambiq",
            dir=params.job_dir,
        )
        wandb.config.update(params.model_dump())
    # END IF

    feat_shape = (params.frame_size, 1)

    datasets = [DatasetFactory.get(ds.name)(**ds.params) for ds in params.datasets]

    train_ds, val_ds = load_train_datasets(
        datasets=datasets,
        params=params,
    )

    # Save validation data
    if params.val_file:
        logger.info(f"Saving validation dataset to {params.val_file}")
        os.makedirs(params.val_file, exist_ok=True)
        val_ds.save(str(params.val_file))

    inputs = keras.Input(shape=feat_shape, name="input", dtype="float32")
    if params.resume and params.model_file:
        logger.debug(f"Loading model from file {params.model_file}")
        model = helia.models.load_model(params.model_file)
        params.model_file = None
    else:
        logger.debug("Creating model from scratch")
        model = ModelFactory.get(params.architecture.name)(
            inputs=inputs,
            params=params.architecture.params,
            num_classes=params.num_classes,
        )
    # END IF

    t_mul = 1
    first_steps = (params.steps_per_epoch * params.epochs) / (np.power(params.lr_cycles, t_mul) - t_mul + 1)
    scheduler = keras.optimizers.schedules.CosineDecayRestarts(
        initial_learning_rate=params.lr_rate,
        first_decay_steps=np.ceil(first_steps),
        t_mul=t_mul,
        m_mul=0.5,
    )

    optimizer = keras.optimizers.Adam(scheduler)
    loss = keras.losses.MeanSquaredError()

    metrics = [
        keras.metrics.MeanAbsoluteError(name="mae"),
        keras.metrics.MeanSquaredError(name="mse"),
        keras.metrics.CosineSimilarity(name="cosine"),
    ]

    if params.resume and params.weights_file:
        logger.debug(f"Hydrating model weights from file {params.weights_file}")
        model.load_weights(params.weights_file)

    if params.model_file is None:
        params.model_file = params.job_dir / "model.keras"

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    flops = helia.metrics.flops.get_flops(model, batch_size=1, fpath=params.job_dir / "model_flops.log")
    model(inputs)
    model.summary(print_fn=logger.info)
    logger.debug(f"Model requires {flops / 1e6:0.2f} MFLOPS")

    ModelCheckpoint = keras.callbacks.ModelCheckpoint
    if helia.utils.env_flag("WANDB"):
        ModelCheckpoint = WandbModelCheckpoint
    model_callbacks = [
        keras.callbacks.EarlyStopping(
            monitor=f"val_{params.val_metric}",
            patience=max(int(0.25 * params.epochs), 1),
            mode="max" if params.val_metric == "f1" else "auto",
            restore_best_weights=True,
            verbose=max(0, params.verbose - 1),
        ),
        ModelCheckpoint(
            filepath=str(params.model_file),
            monitor=f"val_{params.val_metric}",
            save_best_only=True,
            save_weights_only=False,
            mode="max" if params.val_metric == "f1" else "auto",
            verbose=max(0, params.verbose - 1),
        ),
        keras.callbacks.CSVLogger(params.job_dir / "history.csv"),
    ]
    if helia.utils.env_flag("TENSORBOARD"):
        model_callbacks.append(
            keras.callbacks.TensorBoard(
                log_dir=params.job_dir,
                write_steps_per_second=True,
            )
        )
    if helia.utils.env_flag("WANDB"):
        model_callbacks.append(WandbMetricsLogger())

    try:
        model.fit(
            train_ds,
            steps_per_epoch=params.steps_per_epoch,
            verbose=max(0, params.verbose - 1),
            epochs=params.epochs,
            validation_data=val_ds,
            callbacks=model_callbacks,
        )
    except KeyboardInterrupt:
        logger.warning("Stopping training due to keyboard interrupt")

    logger.debug(f"Model saved to {params.model_file}")

    # Get full validation results
    logger.debug("Performing full validation")
