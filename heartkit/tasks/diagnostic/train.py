import logging
import os

import keras
import numpy as np
import pandas as pd
import tensorflow as tf
import wandb
from sklearn.metrics import classification_report, f1_score
from wandb.keras import WandbMetricsLogger, WandbModelCheckpoint

import keras_edge as kedge
from ...defines import HKTrainParams
from ...utils import env_flag, set_random_seed, setup_logger
from ..utils import load_datasets
from .datasets import load_train_datasets
from .utils import create_model

logger = setup_logger(__name__)


def train(params: HKTrainParams):
    """Train  model

    Args:
        params (HKTrainParams): Training parameters
    """
    params.threshold = params.threshold or 0.5

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

    # classes = sorted(list(set(params.class_map.values())))
    class_names = params.class_names or [f"Class {i}" for i in range(params.num_classes)]

    feat_shape = (params.frame_size, 1)
    class_shape = (params.num_classes,)

    ds_spec = (
        tf.TensorSpec(shape=feat_shape, dtype="float32"),
        tf.TensorSpec(shape=class_shape, dtype="int32"),
    )

    datasets = load_datasets(datasets=params.datasets)

    train_ds, val_ds = load_train_datasets(
        datasets=datasets,
        params=params,
        ds_spec=ds_spec,
    )

    test_labels = np.array([label.numpy() for _, label in val_ds])
    y_true = np.concatenate(test_labels)

    class_weights = 0.25
    if params.class_weights == "balanced":
        n_samples = np.sum(y_true)
        class_weights = n_samples / (params.num_classes * np.sum(y_true, axis=0))
        class_weights = (class_weights + class_weights.mean()) / 2  # Smooth out
    # END IF
    logger.info(f"Class weights: {class_weights}")

    inputs = keras.Input(
        shape=ds_spec[0].shape,
        batch_size=None,
        name="input",
        dtype=ds_spec[0].dtype.name,
    )

    if params.resume and params.model_file:
        logger.info(f"Loading model from file {params.model_file}")
        model = kedge.models.load_model(params.model_file)
        params.model_file = None
    else:
        logger.info("Creating model from scratch")
        model = create_model(
            inputs,
            num_classes=params.num_classes,
            architecture=params.architecture,
        )
    # END IF

    flops = kedge.metrics.flops.get_flops(model, batch_size=1, fpath=params.job_dir / "model_flops.log")

    if params.lr_cycles > 1:
        scheduler = keras.optimizers.schedules.CosineDecayRestarts(
            initial_learning_rate=params.lr_rate,
            first_decay_steps=int(0.1 * params.steps_per_epoch * params.epochs),
            t_mul=1.65 / (0.1 * params.lr_cycles * (params.lr_cycles - 1)),
            m_mul=0.4,
        )
    else:
        scheduler = keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=params.lr_rate,
            decay_steps=params.steps_per_epoch * params.epochs,
        )
    # END IF
    optimizer = keras.optimizers.Adam(scheduler)
    loss = keras.losses.BinaryCrossentropy(from_logits=True, label_smoothing=params.label_smoothing)
    # loss = keras.losses.BinaryFocalCrossentropy(
    #     apply_class_balancing=False,
    #     alpha=class_weights,
    #     from_logits=True,
    #     label_smoothing=params.label_smoothing,
    # )
    metrics = [
        keras.metrics.BinaryAccuracy(name="acc"),
        # tfa.MultiF1Score(name="f1", average="weighted"),
    ]

    if params.resume and params.weights_file:
        logger.info(f"Hydrating model weights from file {params.weights_file}")
        model.load_weights(params.weights_file)

    if params.model_file is None:
        params.model_file = params.job_dir / "model.keras"

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
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
    y_pred = model.predict(val_ds)
    y_pred = y_pred >= params.threshold
    cm_path = params.job_dir / "confusion_matrix.png"

    kedge.plotting.cm.multilabel_confusion_matrix_plot(
        y_true=y_true,
        y_pred=y_pred,
        labels=class_names,
        save_path=cm_path,
        normalize="true",
        max_cols=3,
    )

    # Summarize results
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    df_report = pd.DataFrame(report).transpose()
    df_report.to_csv(params.job_dir / "classification_report.csv")
    test_acc = np.sum(y_pred == y_true) / y_true.size
    test_f1 = f1_score(y_true, y_pred, average="weighted")
    logger.info(f"[VAL SET] ACC={test_acc:.2%}, F1={test_f1:.2%}")
