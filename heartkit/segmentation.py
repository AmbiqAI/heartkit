import os
import shutil

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import wandb
from rich.console import Console
from wandb.keras import WandbCallback

from neuralspot.tflite.convert import convert_tflite, predict_tflite, xxd_c_dump
from neuralspot.tflite.metrics import get_flops
from neuralspot.tflite.model import get_strategy, load_model

from .datasets.ludb import LudbDataset
from .datasets.synthetic import SyntheticDataset
from .defines import (
    HeartExportParams,
    HeartSegment,
    HeartTask,
    HeartTestParams,
    HeartTrainParams,
)
from .models.optimizers import Adam
from .tasks import create_task_model, get_num_classes, get_task_shape
from .utils import env_flag, set_random_seed, setup_logger

console = Console()
logger = setup_logger(__name__)


def train_model(params: HeartTrainParams):
    """Train segmentation model.

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
        wandb.init(project=f"ecg-{HeartTask.segmentation}", entity="ambiq", dir=str(params.job_dir))
        wandb.config.update(params.dict())

    datasets = [
        SyntheticDataset(str(params.ds_path), task=HeartTask.segmentation, frame_size=params.frame_size, num_pts=10000),
        # LudbDataset(str(params.ds_path), task=HeartTask.segmentation, frame_size=params.frame_size)
    ]
    train_datasets = []
    val_datasets = []
    for ds in datasets:
        # Create TF datasets
        train_ds, val_ds = ds.load_train_datasets(
            train_patients=params.train_patients,
            val_patients=params.val_patients,
            train_pt_samples=params.samples_per_patient,
            val_pt_samples=params.val_samples_per_patient,
            val_file=params.val_file,
            val_size=params.val_size,
            num_workers=params.data_parallelism,
        )
        train_datasets.append(train_ds)
        val_datasets.append(val_ds)
    # END FOR
    train_ds = tf.data.Dataset.sample_from_datasets(train_datasets)
    # val_ds = tf.data.Dataset.sample_from_datasets(val_datasets)
    train_ds = train_datasets[-1]
    val_ds = val_datasets[-1]

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

    total_steps = params.steps_per_epoch * params.epochs
    lr_scheduler = tf.keras.optimizers.schedules.CosineDecayRestarts(
        initial_learning_rate=1e-4,
        first_decay_steps=int(0.1 * total_steps),
        t_mul=1.65 / (0.1 * 3 * (3 - 1)),
        m_mul=0.4,
    )

    strategy = get_strategy()
    with strategy.scope():
        logger.info("Building model")
        in_shape, _ = get_task_shape(HeartTask.segmentation, params.frame_size)
        inputs = tf.keras.Input(shape=in_shape, batch_size=None, dtype=tf.float32)
        model = create_task_model(inputs, HeartTask.segmentation, params.arch, stages=params.stages)
        flops = get_flops(model, batch_size=1)
        optimizer = Adam(lr_scheduler, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
        loss_fn = tfa.losses.SigmoidFocalCrossEntropy(from_logits=True)
        metrics = [
            tf.keras.metrics.CategoricalAccuracy(name="acc"),
            tf.keras.metrics.OneHotIoU(
                num_classes=get_num_classes(HeartTask.segmentation),
                target_class_ids=(HeartSegment.normal, HeartSegment.pwave, HeartSegment.qrs, HeartSegment.twave),
                name="iou",
            ),
        ]
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
    # END WITH


def evaluate_model(params: HeartTestParams):
    """Test segmentation model.

    Args:
        params (HeartTestParams): Testing/evaluation parameters
    """
    params.seed = set_random_seed(params.seed)
    logger.info(f"Random seed {params.seed}")

    with console.status("[bold green] Loading test dataset..."):
        ds = LudbDataset(str(params.ds_path), task=HeartTask.segmentation, frame_size=params.frame_size)
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
        test_iou = -1
        logger.info(f"[TEST SET] ACC={test_acc:.2%}, IoU={test_iou:.2%}")
    # END WITH


def export_model(params: HeartExportParams):
    """Export segmentation model.

    Args:
        params (HeartDemoParams): Deployment parameters
    """
    tfl_model_path = str(params.job_dir / "model.tflite")
    tflm_model_path = str(params.job_dir / "model_buffer.h")

    # Load model and set fixed batch size of 1
    logger.info("Loading trained model")
    model = load_model(str(params.model_file))
    in_shape, _ = get_task_shape(HeartTask.segmentation, params.frame_size)
    inputs = tf.keras.layers.Input(in_shape, dtype=tf.float32, batch_size=1)
    model(inputs)

    # Load dataset
    with console.status("[bold green] Loading test dataset..."):
        ds = LudbDataset(
            ds_path=str(params.ds_path),
            task=HeartTask.segmentation,
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
    tf_iou = -1
    tfl_acc = np.sum(y_pred_tfl == y_true) / len(y_true)
    tfl_iou = -1
    logger.info(f"[TEST SET]  TF: ACC={tf_acc:.2%}, F1={tf_iou:.2%}")
    logger.info(f"[TEST SET] TFL: ACC={tfl_acc:.2%}, F1={tfl_iou:.2%}")

    # Check accuracy hit
    acc_diff = tf_acc - tfl_acc
    if acc_diff > 0.5:
        logger.warning(f"TFLite accuracy dropped by {100*acc_diff:0.2f}%")
    else:
        logger.info("Validation passed")

    if params.tflm_file and tflm_model_path != params.tflm_file:
        logger.info(f"Copying TFLM header to {params.tflm_file}")
        shutil.copyfile(tflm_model_path, params.tflm_file)


if __name__ == "__main__":
    num_pts = 10000 + 0  # Synthetic + LUDB
    num_datasets = 1
    batch_size = 64
    samples_per_patient = 200
    val_samples_per_patient = 100
    val_patients = 0.20
    val_size = int(val_patients * num_pts) * (val_samples_per_patient - 20)
    steps_per_epoch = int((int((1 - val_patients) * num_pts) * samples_per_patient) / batch_size)
    train_model(
        HeartTrainParams(
            task=HeartTask.segmentation,
            job_dir="./results/segmentation",
            ds_path="./datasets",
            frame_size=1248,
            samples_per_patient=samples_per_patient,
            val_samples_per_patient=val_samples_per_patient,
            val_patients=val_patients,
            val_size=int(val_size / num_datasets),
            batch_size=batch_size,
            buffer_size=8092,
            epochs=80,
            steps_per_epoch=steps_per_epoch,
            val_metric="loss",
        )
    )
