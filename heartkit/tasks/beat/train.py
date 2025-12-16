import os

import keras
import helia_edge as helia
import numpy as np
import sklearn.utils
import wandb
from wandb.integration.keras import WandbMetricsLogger, WandbModelCheckpoint

from ...defines import HKTaskParams
from ...datasets import DatasetFactory
from ...models import ModelFactory
from .datasets import load_train_datasets
from ...utils import setup_plotting


def train(params: HKTaskParams):
    """Train beat task model with given parameters.

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
        wandb.init(project=params.project, entity="ambiq", dir=params.job_dir)
        wandb.config.update(params.model_dump())
    # END IF

    classes = sorted(set(params.class_map.values()))
    class_names = params.class_names or [f"Class {i}" for i in range(params.num_classes)]

    feat_shape = (params.frame_size, 1)

    datasets = [DatasetFactory.get(ds.name)(**ds.params) for ds in params.datasets]

    train_ds, val_ds = load_train_datasets(datasets=datasets, params=params)

    y_true = np.concatenate([y for _, y in val_ds.as_numpy_iterator()])
    y_true = np.argmax(y_true, axis=-1)

    # Save validation data
    if params.val_file:
        logger.info(f"Saving validation dataset to {params.val_file}")
        os.makedirs(params.val_file, exist_ok=True)
        val_ds.save(str(params.val_file))

    class_weights = 0.25
    if params.class_weights == "balanced":
        class_weights = sklearn.utils.compute_class_weight("balanced", classes=np.array(classes), y=y_true)
        class_weights = (class_weights + class_weights.mean()) / 2  # Smooth out
        class_weights = class_weights.tolist()
    # END IF
    logger.debug(f"Class weights: {class_weights}")

    inputs = keras.Input(shape=feat_shape, name="input", dtype="float32")

    if params.resume and params.model_file:
        logger.debug(f"Loading model from file {params.model_file}")
        model = keras.models.load_model(params.model_file)
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

    if params.resume and params.weights_file:
        logger.debug(f"Hydrating model weights from file {params.weights_file}")
        model.load_weights(params.weights_file)

    if params.model_file is None:
        params.model_file = params.job_dir / "model.keras"

    optimizer = keras.optimizers.Adam(scheduler)
    loss = keras.losses.CategoricalFocalCrossentropy(from_logits=True, alpha=class_weights)
    metrics = [
        keras.metrics.CategoricalAccuracy(name="acc"),
        keras.metrics.F1Score(name="f1", average="weighted"),
    ]

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    flops = helia.metrics.flops.get_flops(model, batch_size=1, fpath=params.job_dir / "model_flops.log")
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
            metrics=["loss", "acc"],
            save_path=params.job_dir / "history.png",
            title="Training History",
            stack=True,
            figsize=(9, 5),
        )

    # Get full validation results
    logger.debug("Performing full validation")
    y_pred = np.argmax(model.predict(val_ds, verbose=params.verbose), axis=-1)

    cm_path = params.job_dir / "confusion_matrix.png"
    helia.plotting.confusion_matrix_plot(y_true, y_pred, labels=class_names, save_path=cm_path, normalize="true")
    helia.plotting.px_plot_confusion_matrix(
        y_true,
        y_pred,
        labels=class_names,
        save_path=cm_path.with_suffix(".html"),
        normalize="true",
    )

    # Summarize results
    rst = model.evaluate(val_ds, verbose=params.verbose, return_dict=True)
    logger.info("[VAL SET] " + ", ".join([f"{k}={v:0.4f}" for k, v in rst.items()]))

    # cleanup
    keras.utils.clear_session()
    for ds in datasets:
        ds.close()
