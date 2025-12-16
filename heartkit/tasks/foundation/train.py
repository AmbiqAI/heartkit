import os

import keras
import wandb
import numpy as np
from wandb.integration.keras import WandbMetricsLogger, WandbModelCheckpoint
import helia_edge as helia

from ...defines import HKTaskParams
from ...models import ModelFactory
from ...datasets import DatasetFactory
from .datasets import load_train_datasets
from ...utils import setup_plotting


def train(params: HKTaskParams):
    """Train model for foundation task using SimCLR

    Args:
        params (HKTaskParams): Task parameters
    """
    os.makedirs(params.job_dir, exist_ok=True)
    logger = helia.utils.setup_logger(__name__, level=params.verbose, file_path=params.job_dir / "train.log")
    logger.debug(f"Creating working directory in {params.job_dir}")

    params.temperature = float(getattr(params, "temperature", 0.1))

    params.seed = helia.utils.set_random_seed(params.seed)
    logger.debug(f"Random seed {params.seed}")

    with open(params.job_dir / "configuration.json", "w", encoding="utf-8") as fp:
        fp.write(params.model_dump_json(indent=2))

    if helia.utils.env_flag("WANDB"):
        wandb.init(project=params.project, entity="ambiq", dir=params.job_dir)
        wandb.config.update(params.model_dump())
    # END IF

    feat_shape = (params.frame_size, 1)

    datasets = [DatasetFactory.get(ds.name)(**ds.params) for ds in params.datasets]

    train_ds, val_ds = load_train_datasets(datasets=datasets, params=params)

    # Save validation data
    if params.val_file:
        logger.info(f"Saving validation dataset to {params.val_file}")
        os.makedirs(params.val_file, exist_ok=True)
        val_ds.save(str(params.val_file))

    # Create encoder
    encoder_input = keras.Input(shape=feat_shape, dtype="float32")
    encoder = ModelFactory.get(params.architecture.name)(
        inputs=encoder_input,
        params=params.architecture.params,
        num_classes=None,
    )

    flops = helia.metrics.flops.get_flops(encoder, batch_size=1, fpath=params.job_dir / "encoder_flops.log")
    encoder.summary(print_fn=logger.info)
    logger.debug(f"Encoder requires {flops / 1e6:0.2f} MFLOPS")

    # Create  projector
    # encoder_output = encoder(encoder_input)
    # projection_width = params.num_classes
    # projector_input = encoder_output
    # projector_output = keras.layers.Dense(projection_width, activation="relu6")(projector_input)
    # projector_output = keras.layers.Dense(projection_width)(projector_output)
    # projector = keras.Model(inputs=projector_input, outputs=projector_output, name="projector")
    # flops = helia.metrics.flops.get_flops(projector, batch_size=1, fpath=params.job_dir / "projector_flops.log")
    # projector.summary(print_fn=logger.info)
    # logger.debug(f"Projector requires {flops/1e6:0.2f} MFLOPS")

    if params.model_file is None:
        params.model_file = params.job_dir / "model.keras"

    model = helia.trainers.SimCLRTrainer(
        encoder=encoder,
        projector=None,
    )

    def get_scheduler():
        t_mul = 1
        first_steps = (params.steps_per_epoch * params.epochs) / (np.power(params.lr_cycles, t_mul) - t_mul + 1)
        scheduler = keras.optimizers.schedules.CosineDecayRestarts(
            initial_learning_rate=params.lr_rate,
            first_decay_steps=np.ceil(first_steps),
            t_mul=t_mul,
            m_mul=0.5,
        )
        return scheduler

    model.compile(
        encoder_optimizer=keras.optimizers.Adam(get_scheduler()),
        encoder_loss=helia.losses.simclr.SimCLRLoss(temperature=params.temperature),
        encoder_metrics=[keras.metrics.MeanSquaredError(name="mse"), keras.metrics.CosineSimilarity(name="cos")],
    )

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
    # Use minimal progress bar
    if params.verbose <= 1:
        model_callbacks.append(
            helia.callbacks.TQDMProgressBar(
                show_epoch_progress=False,
            )
        )
    try:
        history = model.fit(
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

    setup_plotting()
    if history:
        helia.plotting.plot_history_metrics(
            history.history,
            metrics=["loss", "cos"],
            save_path=params.job_dir / "history.png",
            title="Training History",
            stack=True,
            figsize=(9, 5),
        )

    metrics = model.evaluate(val_ds, verbose=2, return_dict=True)
    logger.info("[VAL SET] " + ", ".join(f"{k.upper()}: {v:.4f}" for k, v in metrics.items()))

    # cleanup
    keras.utils.clear_session()
    for ds in datasets:
        ds.close()
