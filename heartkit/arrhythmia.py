import os
import shutil

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import wandb
from rich.console import Console
from sklearn.metrics import f1_score
from wandb.keras import WandbCallback

from neuralspot.tflite.convert import convert_tflite, predict_tflite, xxd_c_dump
from neuralspot.tflite.metrics import get_flops
from neuralspot.tflite.model import get_strategy, load_model

from .datasets import IcentiaDataset
from .defines import HeartExportParams, HeartTask, HeartTestParams, HeartTrainParams
from .metrics import confusion_matrix_plot, roc_auc_plot
from .models.optimizers import Adam
from .models.utils import get_predicted_threshold_indices
from .tasks import create_task_model, get_class_names, get_task_shape
from .utils import env_flag, set_random_seed, setup_logger

console = Console()
logger = setup_logger(__name__)


def train_model(params: HeartTrainParams):
    """Train rhythm-level arrhythmia model.

    Args:
        params (HeartTrainParams): Training parameters
    """

    params.seed = set_random_seed(params.seed)
    logger.info(f"Random seed {params.seed}")

    os.makedirs(str(params.job_dir), exist_ok=True)
    logger.info(f"Creating working directory in {params.job_dir}")
    with open(str(params.job_dir / "train_config.json"), "w", encoding="utf-8") as fp:
        fp.write(params.json(indent=2))

    if env_flag("WANDB"):
        wandb.init(project=f"ecg-{HeartTask.rhythm}", entity="ambiq", dir=str(params.job_dir))
        wandb.config.update(params.dict())

    # Create TF datasets
    ds = IcentiaDataset(
        ds_path=str(params.ds_path),
        task=HeartTask.rhythm,
        frame_size=params.frame_size,
    )
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
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )
    val_ds = val_ds.batch(
        batch_size=params.batch_size,
        drop_remainder=True,
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    strategy = get_strategy()
    with strategy.scope():
        logger.info("Building model")
        in_shape, _ = get_task_shape(HeartTask.rhythm, params.frame_size)
        inputs = tf.keras.Input(in_shape, batch_size=None, dtype=tf.float32)
        model = create_task_model(inputs, HeartTask.rhythm, params.arch, stages=params.stages)
        flops = get_flops(model, batch_size=1)
        optimizer = Adam(
            tf.keras.optimizers.schedules.CosineDecayRestarts(
                initial_learning_rate=1e-3,
                first_decay_steps=int(0.1 * params.steps_per_epoch * params.epochs),
                t_mul=1.8 / (0.1 * 3 * (3 - 1)),  # 3 cycles
                m_mul=0.40,
            ),
            beta_1=0.9,
            beta_2=0.98,
            epsilon=1e-9,
        )
        loss_fn = tfa.losses.SigmoidFocalCrossEntropy(from_logits=True)
        metrics = [tf.keras.metrics.CategoricalAccuracy(name="acc")]
        model.compile(optimizer=optimizer, loss=loss_fn, metrics=metrics)
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
            tf.keras.callbacks.CSVLogger(str(params.job_dir / "history.csv")),
            tf.keras.callbacks.TensorBoard(log_dir=str(params.job_dir), write_steps_per_second=True),
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
        tf_model_path = str(params.job_dir / "model.tf")
        logger.info(f"Model saved to {tf_model_path}")
        model.save(tf_model_path)

        # Get full validation results
        logger.info("Performing full validation")
        test_labels = [label.numpy() for _, label in val_ds]
        y_true = np.argmax(np.concatenate(test_labels), axis=1)
        y_pred = np.argmax(model.predict(val_ds), axis=1)

        # Summarize results
        class_names = get_class_names(task=HeartTask.rhythm)
        test_acc = np.sum(y_pred == y_true) / len(y_true)
        test_f1 = f1_score(y_true, y_pred, average="macro")
        logger.info(f"[VAL SET] ACC={test_acc:.2%}, F1={test_f1:.2%}")
        cm_path = str(params.job_dir / "confusion_matrix.png")
        confusion_matrix_plot(y_true, y_pred, labels=class_names, save_path=cm_path)
        if env_flag("WANDB"):
            wandb.log(
                {
                    "conf_mat": wandb.plot.confusion_matrix(
                        probs=None, preds=y_pred, y_true=y_true, class_names=class_names
                    )
                }
            )
        # END IF
    # END WITH


def evaluate_model(params: HeartTestParams):
    """Test arrhythmia model.

    Args:
        params (HeartTestParams): Testing/evaluation parameters
    """
    params.seed = set_random_seed(params.seed)
    logger.info(f"Random seed {params.seed}")

    with console.status("[bold green] Loading test dataset..."):
        ds = IcentiaDataset(
            ds_path=str(params.ds_path),
            task=HeartTask.rhythm,
            frame_size=params.frame_size,
        )
        test_ds = ds.load_test_dataset(
            test_patients=params.test_patients,
            test_pt_samples=params.samples_per_patient,
            num_workers=params.data_parallelism,
        )
        test_x, test_y = next(test_ds.batch(params.test_size).as_numpy_iterator())
    # END WITH

    strategy = get_strategy()
    with strategy.scope():
        logger.info("Loading model")
        model = load_model(str(params.model_file))
        model.summary()

        logger.info("Performing inference")
        y_true = np.argmax(test_y, axis=1)
        y_prob = tf.nn.softmax(model.predict(test_x)).numpy()
        y_pred = np.argmax(y_prob, axis=1)

        # Summarize results
        logger.info("Testing Results")
        test_acc = np.sum(y_pred == y_true) / len(y_true)
        test_f1 = f1_score(y_true, y_pred, average="macro")
        logger.info(f"[TEST SET] ACC={test_acc:.2%}, F1={test_f1:.2%}")

        # If threshold given, only count predictions above threshold
        if params.threshold is not None:
            y_thresh_idx = get_predicted_threshold_indices(y_prob, y_pred, params.threshold)
            drop_perc = 1 - len(y_thresh_idx) / len(y_true)
            y_prob = y_prob[y_thresh_idx]
            y_pred = y_pred[y_thresh_idx]
            y_true = y_true[y_thresh_idx]
            test_acc = np.sum(y_pred == y_true) / len(y_true)
            test_f1 = f1_score(y_true, y_pred, average="macro")
            logger.info(
                f"[TEST SET] ACC={test_acc:.2%}, F1={test_f1:.2%}, THRESH={params.threshold:0.2%}, DROP={drop_perc:.2%}"
            )
        # END IF
        cm_path = str(params.job_dir / "confusion_matrix_test.png")
        class_names = get_class_names(HeartTask.rhythm)
        confusion_matrix_plot(y_true, y_pred, labels=class_names, save_path=cm_path)
        if len(class_names) == 2:
            roc_path = str(params.job_dir / "roc_auc_test.png")
            roc_auc_plot(y_true, y_prob[:, 1], labels=class_names, save_path=roc_path)
        # END IF
    # END WITH


def export_model(params: HeartExportParams):
    """Export arrhythmia model.

    Args:
        params (HeartDemoParams): Deployment parameters
    """
    tfl_model_path = str(params.job_dir / "model.tflite")
    tflm_model_path = str(params.job_dir / "model_buffer.h")

    # Load model and set fixed batch size of 1
    logger.info("Loading trained model")
    model = load_model(str(params.model_file))
    in_shape, _ = get_task_shape(HeartTask.rhythm, params.frame_size)
    inputs = tf.keras.layers.Input(in_shape, dtype=tf.float32, batch_size=1)
    model(inputs)

    # Load dataset
    with console.status("[bold green] Loading test dataset..."):
        ds = IcentiaDataset(
            ds_path=str(params.ds_path),
            task=HeartTask.rhythm,
            frame_size=params.frame_size,
        )
        test_ds = ds.load_test_dataset(
            test_pt_samples=params.samples_per_patient,
            num_workers=params.data_parallelism,
        )
        test_x, test_y = next(test_ds.batch(params.test_size).as_numpy_iterator())
    # END WITH

    logger.info("Converting model to TFLite")
    tflite_model = convert_tflite(
        model,
        quantize=params.quantization,
        test_x=test_x[:1000],
        input_type=tf.int8 if params.quantization else None,
        output_type=tf.int8 if params.quantization else None,
    )

    # Save TFLite model
    logger.info(f"Saving TFLite model to {tfl_model_path}")
    with open(tfl_model_path, "wb") as fp:
        fp.write(tflite_model)

    # Save TFLM model
    logger.info(f"Saving TFL micro model to {tflm_model_path}")
    xxd_c_dump(
        src_path=tfl_model_path,
        dst_path=tflm_model_path,
        var_name=params.tflm_var_name,
        chunk_len=20,
        is_header=True,
    )

    # Verify TFLite results match TF results on example data
    logger.info("Validating model results")
    y_true = np.argmax(test_y, axis=1)
    y_prob_tf = tf.nn.softmax(model.predict(test_x)).numpy()
    y_pred_tf = np.argmax(y_prob_tf, axis=1)
    y_prob_tfl = tf.nn.softmax(predict_tflite(model_content=tflite_model, test_x=test_x)).numpy()
    y_pred_tfl = np.argmax(y_prob_tfl, axis=1)

    tf_acc = np.sum(y_pred_tf == y_true) / len(y_true)
    tf_f1 = f1_score(y_true, y_pred_tf, average="macro")
    tfl_acc = np.sum(y_pred_tfl == y_true) / len(y_true)
    tfl_f1 = f1_score(y_true, y_pred_tfl, average="macro")
    logger.info(f"[TEST SET]  TF: ACC={tf_acc:.2%}, F1={tf_f1:.2%}")
    logger.info(f"[TEST SET] TFL: ACC={tfl_acc:.2%}, F1={tfl_f1:.2%}")

    if params.threshold is not None:
        y_thresh_idx = np.union1d(
            get_predicted_threshold_indices(y_prob_tf, y_pred_tf, params.threshold),
            get_predicted_threshold_indices(y_prob_tfl, y_pred_tfl, params.threshold),
        )
        y_thresh_idx.sort()
        drop_perc = 1 - len(y_thresh_idx) / len(y_true)
        y_pred_tf = y_pred_tf[y_thresh_idx]
        y_pred_tfl = y_pred_tfl[y_thresh_idx]
        y_true = y_true[y_thresh_idx]

        tf_acc = np.sum(y_pred_tf == y_true) / len(y_true)
        tf_f1 = f1_score(y_true, y_pred_tf, average="macro")
        tfl_acc = np.sum(y_pred_tfl == y_true) / len(y_true)
        tfl_f1 = f1_score(y_true, y_pred_tfl, average="macro")

        logger.info(
            f"[TEST SET]  TF: ACC={tf_acc:.2%}, F1={tf_f1:.2%}, THRESH={params.threshold:0.2%}, DROP={drop_perc:.2%}"
        )
        logger.info(
            f"[TEST SET] TFL: ACC={tfl_acc:.2%}, F1={tfl_f1:.2%}, THRESH={params.threshold:0.2%}, DROP={drop_perc:.2%}"
        )
    # END IF

    # Check accuracy hit
    acc_diff = tf_acc - tfl_acc
    if acc_diff > 0.5:
        logger.warning(f"TFLite accuracy dropped by {100*acc_diff:0.2f}%")
    else:
        logger.info("Validation passed")

    if params.tflm_file and tflm_model_path != params.tflm_file:
        logger.info(f"Copying TFLM header to {params.tflm_file}")
        shutil.copyfile(tflm_model_path, params.tflm_file)
