import logging
import os

import numpy as np
import sklearn.utils
import tensorflow as tf
import tensorflow_model_optimization as tfmot
import wandb
from wandb.keras import WandbCallback

from .. import tflite as tfa
from ..defines import HeartTask, HeartTrainParams
from ..metrics import compute_iou, confusion_matrix_plot, f1_score
from ..utils import env_flag, set_random_seed, setup_logger
from .defines import get_class_mapping, get_class_names, get_classes
from .utils import create_model, load_datasets, load_train_datasets

logger = setup_logger(__name__)


def train(params: HeartTrainParams):
    """Train segmentation model.

    Args:
        params (HeartTrainParams): Training parameters
    """

    params.finetune = bool(getattr(params, "finetune", False))
    params.lr_rate = getattr(params, "lr_rate", 1e-4)
    params.lr_cycles = int(getattr(params, "lr_cycles", 3))
    params.datasets = getattr(params, "datasets", ["ludb"])
    params.num_pts = getattr(params, "num_pts", 1000)
    params.steps_per_epoch = params.steps_per_epoch or 100
    params.seed = set_random_seed(params.seed)
    logger.info(f"Random seed {params.seed}")

    os.makedirs(params.job_dir, exist_ok=True)
    logger.info(f"Creating working directory in {params.job_dir}")
    with open(params.job_dir / "train_config.json", "w", encoding="utf-8") as fp:
        fp.write(params.json(indent=2))

    handler = logging.FileHandler(params.job_dir / "train.log", mode="w")
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)

    if env_flag("WANDB"):
        wandb.init(
            project=f"heartkit-{HeartTask.segmentation}",
            entity="ambiq",
            dir=params.job_dir,
        )
        wandb.config.update(params.dict())

    classes = get_classes(params.num_classes)
    class_names = get_class_names(params.num_classes)
    class_map = get_class_mapping(params.num_classes)

    datasets = load_datasets(
        ds_path=params.ds_path,
        frame_size=params.frame_size,
        sampling_rate=params.sampling_rate,
        class_map=class_map,
        dataset_names=params.datasets,
        num_pts=params.num_pts,
    )
    train_ds, val_ds = load_train_datasets(datasets, params)

    test_labels = [y.numpy() for _, y in val_ds]
    y_true = np.argmax(np.concatenate(test_labels).squeeze(), axis=-1).flatten()

    class_weights = sklearn.utils.compute_class_weight("balanced", classes=classes, y=y_true)
    class_weights = 0.25

    with tfa.get_strategy().scope():
        logger.info("Building model")
        inputs = tf.keras.Input(shape=datasets[0].feat_shape, batch_size=None, dtype=tf.float32)
        if params.model_file:
            model = tfa.load_model(params.model_file)
        else:
            model = create_model(
                inputs,
                num_classes=len(classes),
                name=params.model,
                params=params.model_params,
            )

        # If fine-tune, freeze model encoder weights
        if params.finetune:
            for layer in model.layers:
                if layer.name.startswith("ENC"):
                    logger.info(f"Freezing {layer.name}")
                    layer.trainable = False
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
        optimizer = tf.keras.optimizers.Adam(scheduler, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
        loss = tf.keras.losses.CategoricalFocalCrossentropy(
            from_logits=True,
            alpha=class_weights,
        )
        metrics = [
            tf.keras.metrics.CategoricalAccuracy(name="acc"),
            tf.keras.metrics.OneHotIoU(
                num_classes=len(classes),
                target_class_ids=classes,
                name="iou",
            ),
        ]

        if params.weights_file:
            logger.info(f"Loading weights from file {params.weights_file}")
            model.load_weights(params.weights_file)

        params.weights_file = params.job_dir / "model.weights"

        # Perform QAT if requested (typically used for fine-tuning)
        if params.quantization:
            logger.info("Performing QAT...")
            model = tfmot.quantization.keras.quantize_model(model)

        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        model(inputs)
        model.summary(print_fn=logger.info)
        logger.info(f"Model requires {flops/1e6:0.2f} MFLOPS")

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
        y_pred = np.argmax(model.predict(val_ds).squeeze(), axis=-1).flatten()

        confusion_matrix_plot(
            y_true=y_true,
            y_pred=y_pred,
            labels=class_names,
            save_path=params.job_dir / "confusion_matrix.png",
            normalize="true",
        )
        if env_flag("WANDB"):
            conf_mat = wandb.plot.confusion_matrix(preds=y_pred, y_true=y_true, class_names=class_names)
            wandb.log({"conf_mat": conf_mat})
        # END IF

        # Summarize results
        test_acc = np.sum(y_pred == y_true) / y_true.size
        test_f1 = f1_score(y_true=y_true, y_pred=y_pred, average="weighted")
        test_iou = compute_iou(y_true, y_pred, average="weighted")
        logger.info(f"[TEST SET] ACC={test_acc:.2%}, F1={test_f1:.2%} IoU={test_iou:0.2%}")
    # END WITH
