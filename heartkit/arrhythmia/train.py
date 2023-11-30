import os

import numpy as np
import sklearn.utils
import tensorflow as tf
import wandb
from sklearn.metrics import f1_score
from wandb.keras import WandbCallback

from .. import tflite as tfa
from ..defines import HeartTask, HeartTrainParams
from ..metrics import confusion_matrix_plot
from ..utils import env_flag, set_random_seed, setup_logger
from .defines import get_class_mapping, get_class_names, get_classes
from .utils import create_model, load_dataset, load_train_datasets

logger = setup_logger(__name__)


def train(params: HeartTrainParams):
    """Train rhythm-level arrhythmia model.

    Args:
        params (HeartTrainParams): Training parameters
    """
    params.lr_rate = getattr(params, "lr_rate", 1e-3)
    params.lr_cycles = getattr(params, "lr_cycles", 3)
    params.steps_per_epoch = params.steps_per_epoch or 50
    params.seed = set_random_seed(params.seed)
    logger.info(f"Random seed {params.seed}")

    os.makedirs(params.job_dir, exist_ok=True)
    logger.info(f"Creating working directory in {params.job_dir}")
    with open(params.job_dir / "train_config.json", "w", encoding="utf-8") as fp:
        fp.write(params.json(indent=2))

    if env_flag("WANDB"):
        wandb.init(
            project=f"heartkit-{HeartTask.arrhythmia}",
            entity="ambiq",
            dir=params.job_dir,
        )
        wandb.config.update(params.dict())

    classes = get_classes(params.num_classes)
    class_names = get_class_names(params.num_classes)
    class_map = get_class_mapping(params.num_classes)

    ds = load_dataset(
        ds_path=params.ds_path, frame_size=params.frame_size, sampling_rate=params.sampling_rate, class_map=class_map
    )
    train_ds, val_ds = load_train_datasets(ds, params)

    test_labels = [label.numpy() for _, label in val_ds]
    y_true = np.argmax(np.concatenate(test_labels), axis=1)
    class_weights = sklearn.utils.compute_class_weight("balanced", classes=classes, y=y_true)
    class_weights = 0.25

    with tfa.get_strategy().scope():
        logger.info("Building model")
        inputs = tf.keras.Input(ds.feat_shape, batch_size=None, dtype=tf.float32)
        model = create_model(inputs, num_classes=params.num_classes, name=params.model, params=params.model_params)
        flops = tfa.get_flops(model, batch_size=1, fpath=params.job_dir / "model_flops.log")

        if params.lr_cycles > 1:
            scheduler = tf.keras.optimizers.schedules.CosineDecayRestarts(
                initial_learning_rate=params.lr_rate,
                first_decay_steps=int(0.1 * params.steps_per_epoch * params.epochs),
                t_mul=1.65 / (0.1 * params.lr_cycles * (params.lr_cycles - 1)),
                m_mul=0.4,
            )
        else:
            scheduler = tf.keras.optimizers.schedules.CosineDecay(
                initial_learning_rate=params.lr_rate, decay_steps=params.steps_per_epoch * params.epochs
            )
        optimizer = tf.keras.optimizers.Adam(scheduler)
        loss = tf.keras.losses.CategoricalFocalCrossentropy(from_logits=True, alpha=class_weights)
        metrics = [tf.keras.metrics.CategoricalAccuracy(name="acc")]
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        model(inputs)

        model.summary(print_fn=logger.info)
        logger.info(f"Model requires {flops/1e6:0.2f} MFLOPS")

        if params.weights_file:
            logger.info(f"Loading weights from file {params.weights_file}")
            model.load_weights(params.weights_file)
        params.weights_file = params.job_dir / "model.weights"

        model_callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor=f"val_{params.val_metric}",
                patience=max(int(0.25 * params.epochs), 1),
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
            tf.keras.callbacks.CSVLogger(params.job_dir / "history.csv"),
            tf.keras.callbacks.TensorBoard(log_dir=params.job_dir, write_steps_per_second=True),
        ]
        if env_flag("WANDB"):
            model_callbacks.append(WandbCallback())

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
        tf_model_path = params.job_dir / "model.tf"
        logger.info(f"Model saved to {tf_model_path}")
        model.save(tf_model_path)

        # Get full validation results
        logger.info("Performing full validation")
        y_pred = np.argmax(model.predict(val_ds), axis=1)

        # Summarize results
        test_acc = np.sum(y_pred == y_true) / len(y_true)
        test_f1 = f1_score(y_true, y_pred, average="weighted")
        logger.info(f"[VAL SET] ACC={test_acc:.2%}, F1={test_f1:.2%}")
        cm_path = params.job_dir / "confusion_matrix.png"
        confusion_matrix_plot(y_true, y_pred, labels=class_names, save_path=cm_path, normalize="true")
        if env_flag("WANDB"):
            conf_mat = wandb.plot.confusion_matrix(preds=y_pred, y_true=y_true, class_names=class_names)
            wandb.log({"conf_mat": conf_mat})
        # END IF
    # END WITH
