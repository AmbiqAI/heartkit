import os
import shutil

import numpy as np
import numpy.typing as npt
import tensorflow as tf
import tensorflow_model_optimization as tfmot
import wandb
from rich.console import Console
from wandb.keras import WandbCallback

from neuralspot.tflite.convert import convert_tflite, predict_tflite, xxd_c_dump
from neuralspot.tflite.metrics import get_flops
from neuralspot.tflite.model import get_strategy, load_model

from . import signal
from .datasets import (
    HeartKitDataset,
    LudbDataset,
    QtdbDataset,
    SyntheticDataset,
    augment_pipeline,
)
from .defines import (
    HeartExportParams,
    HeartSegment,
    HeartTask,
    HeartTestParams,
    HeartTrainParams,
)
from .metrics import compute_iou, confusion_matrix_plot
from .tasks import create_task_model, get_class_names, get_num_classes, get_task_shape
from .utils import env_flag, set_random_seed, setup_logger

console = Console()
logger = setup_logger(__name__)


def prepare(x: npt.NDArray, sample_rate: float) -> npt.NDArray:
    """Prepare dataset."""
    x = signal.filter_signal(
        x,
        lowcut=0.5,
        highcut=30,
        order=3,
        sample_rate=sample_rate,
        axis=0,
        forward_backward=True,
    )
    x = signal.normalize_signal(x, eps=0.1, axis=None)
    return x


def load_train_datasets(
    params: HeartTrainParams,
) -> tuple[tf.data.Dataset, tf.data.Dataset]:
    """Load segmentation train datasets.
    Args:
        params (HeartTrainParams): Train params
    Returns:
        tuple[tf.data.Dataset, tf.data.Dataset]: Train and validation datasets
    """
    dataset_names: list[str] = getattr(params, "datasets", ["ludb"])
    num_pts = getattr(params, "num_pts", 1000)
    datasets: list[HeartKitDataset] = []
    if "synthetic" in dataset_names:
        datasets.append(
            SyntheticDataset(
                str(params.ds_path),
                task=HeartTask.segmentation,
                frame_size=params.frame_size,
                target_rate=params.sampling_rate,
                num_pts=num_pts,
            )
        )
    if "ludb" in dataset_names:
        datasets.append(
            LudbDataset(
                str(params.ds_path),
                task=HeartTask.segmentation,
                frame_size=params.frame_size,
                target_rate=params.sampling_rate,
            )
        )
    if "qtdb" in dataset_names:
        datasets.append(
            QtdbDataset(
                str(params.ds_path),
                task=HeartTask.segmentation,
                frame_size=params.frame_size,
                target_rate=params.sampling_rate,
            )
        )

    def preprocess(x: npt.NDArray) -> npt.NDArray:
        xx = x.copy().squeeze()
        if params.augmentations:
            xx = augment_pipeline(xx, augmentations=params.augmentations, sample_rate=params.sampling_rate)
        xx = prepare(xx, sample_rate=params.sampling_rate)
        return xx

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
            preprocess=preprocess,
            num_workers=params.data_parallelism,
        )
        train_datasets.append(train_ds)
        val_datasets.append(val_ds)
    # END FOR
    ds_weights = np.array([len(ds.get_train_patient_ids()) for ds in datasets])
    ds_weights = ds_weights / ds_weights.sum()

    train_ds = tf.data.Dataset.sample_from_datasets(train_datasets, weights=ds_weights)
    val_ds = tf.data.Dataset.sample_from_datasets(val_datasets, weights=ds_weights)

    # Shuffle and batch datasets for training
    train_ds = (
        train_ds.shuffle(
            buffer_size=params.buffer_size,
            reshuffle_each_iteration=True,
        )
        .batch(
            batch_size=params.batch_size,
            drop_remainder=False,
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )
    val_ds = val_ds.batch(
        batch_size=params.batch_size,
        drop_remainder=True,
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    return train_ds, val_ds


def load_test_datasets(
    params: HeartTestParams | HeartExportParams,
) -> tuple[npt.NDArray, npt.NDArray]:
    """Load segmentation test dataset.
    Args:
        params (HeartTestParams|HeartExportParams): Test params
    Returns:
        tuple[npt.NDArray, npt.NDArray]: Test data and labels
    """
    dataset_names: list[str] = getattr(params, "datasets", ["ludb"])

    def preprocess(x: npt.NDArray) -> npt.NDArray:
        xx = x.copy().squeeze()
        xx = prepare(xx, sample_rate=params.sampling_rate)
        return xx

    with console.status("[bold green] Loading test dataset..."):
        datasets: list[HeartKitDataset] = []
        if "synthetic" in dataset_names:
            datasets.append(
                SyntheticDataset(
                    str(params.ds_path),
                    task=HeartTask.segmentation,
                    frame_size=params.frame_size,
                    target_rate=params.sampling_rate,
                    num_pts=getattr(params, "num_pts", 200),
                )
            )
        if "ludb" in dataset_names:
            datasets.append(
                LudbDataset(
                    str(params.ds_path),
                    task=HeartTask.segmentation,
                    frame_size=params.frame_size,
                    target_rate=params.sampling_rate,
                )
            )
        if "qtdb" in dataset_names:
            datasets.append(
                QtdbDataset(
                    str(params.ds_path),
                    task=HeartTask.segmentation,
                    frame_size=params.frame_size,
                    target_rate=params.sampling_rate,
                )
            )

        test_datasets = [
            ds.load_test_dataset(
                test_pt_samples=params.samples_per_patient,
                preprocess=preprocess,
                num_workers=params.data_parallelism,
            )
            for ds in datasets
        ]
        ds_weights = np.array([len(ds.get_test_patient_ids()) for ds in datasets])
        ds_weights = ds_weights / ds_weights.sum()

        test_ds = tf.data.Dataset.sample_from_datasets(test_datasets, weights=ds_weights)
        test_x, test_y = next(test_ds.batch(params.test_size).as_numpy_iterator())
    # END WITH
    return test_x, test_y


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
        wandb.init(
            project=f"heartkit-{HeartTask.segmentation}",
            entity="ambiq",
            dir=params.job_dir,
        )
        wandb.config.update(params.dict())

    train_ds, val_ds = load_train_datasets(params)

    strategy = get_strategy()
    with strategy.scope():
        logger.info("Building model")
        in_shape, _ = get_task_shape(HeartTask.segmentation, params.frame_size)
        inputs = tf.keras.Input(shape=in_shape, batch_size=None, dtype=tf.float32)
        model = create_task_model(
            inputs,
            HeartTask.segmentation,
            name=params.model,
            params=params.model_params,
        )
        # If fine-tune, freeze model encoder weights
        if bool(getattr(params, "finetune", False)):
            for layer in model.layers:
                if layer.name.startswith("ENC"):
                    logger.info(f"Freezing {layer.name}")
                    layer.trainable = False
        flops = get_flops(model, batch_size=1, fpath=str(params.job_dir / "model_flops.log"))

        # Grab optional LR parameters
        lr_rate: float = getattr(params, "lr_rate", 1e-4)
        lr_cycles: int = getattr(params, "lr_cycles", 3)
        steps_per_epoch = params.steps_per_epoch or 1000
        scheduler = tf.keras.optimizers.schedules.CosineDecayRestarts(
            initial_learning_rate=lr_rate,
            first_decay_steps=int(0.1 * steps_per_epoch * params.epochs),
            t_mul=1.65 / (0.1 * lr_cycles * (lr_cycles - 1)),
            m_mul=0.4,
        )
        optimizer = tf.keras.optimizers.Adam(scheduler, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
        loss = tf.keras.losses.CategoricalFocalCrossentropy(from_logits=True)
        metrics = [
            tf.keras.metrics.CategoricalAccuracy(name="acc"),
            tf.keras.metrics.OneHotIoU(
                num_classes=get_num_classes(HeartTask.segmentation),
                target_class_ids=tuple(s.value for s in HeartSegment),
                name="iou",
            ),
        ]
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        model(inputs)

        model.summary(print_fn=logger.info)
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
                steps_per_epoch=steps_per_epoch,
                verbose=2,
                epochs=params.epochs,
                validation_data=val_ds,
                callbacks=model_callbacks,
            )
        except KeyboardInterrupt:
            logger.warning("Stopping training due to keyboard interrupt")

        # Restore best weights from checkpoint
        model.load_weights(params.weights_file)

        if params.quantization:
            logger.info("Performing QAT fine-tuning")
            qmodel = tfmot.quantization.keras.quantize_model(model)
            num_epochs = int(0.25 * params.epochs)
            scheduler = tf.keras.optimizers.schedules.CosineDecay(
                initial_learning_rate=lr_rate / 5, decay_steps=steps_per_epoch * num_epochs
            )
            optimizer = tf.keras.optimizers.Adam(scheduler, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
            loss = tf.keras.losses.CategoricalFocalCrossentropy(from_logits=True)
            metrics = [
                tf.keras.metrics.CategoricalAccuracy(name="acc"),
                tf.keras.metrics.OneHotIoU(
                    num_classes=get_num_classes(HeartTask.segmentation),
                    target_class_ids=tuple(s.value for s in HeartSegment),
                    name="iou",
                ),
            ]
            qmodel.compile(optimizer=optimizer, loss=loss, metrics=metrics)
            qmodel(inputs)
            qmodel.summary(print_fn=logger.info)

            model_callbacks = [
                tf.keras.callbacks.EarlyStopping(
                    monitor=f"val_{params.val_metric}",
                    patience=max(int(0.50 * num_epochs), 1),
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
                tf.keras.callbacks.CSVLogger(str(params.job_dir / "history.csv"), append=True),
                tf.keras.callbacks.TensorBoard(log_dir=str(params.job_dir), write_steps_per_second=True),
            ]
            if env_flag("WANDB"):
                model_callbacks.append(WandbCallback())

            try:
                qmodel.fit(
                    train_ds,
                    steps_per_epoch=steps_per_epoch,
                    verbose=2,
                    initial_epoch=params.epochs,
                    epochs=params.epochs + num_epochs,
                    validation_data=val_ds,
                    callbacks=model_callbacks,
                )
            except KeyboardInterrupt:
                logger.warning("Stopping training due to keyboard interrupt")

            # Restore best weights from checkpoint
            qmodel.load_weights(params.weights_file)
            model = qmodel  # Replace model w/ quantized version
        # END IF

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
    test_x, test_y = load_test_datasets(params)

    with tfmot.quantization.keras.quantize_scope():
        logger.info("Loading model")
        model = load_model(str(params.model_file))
        flops = get_flops(model, batch_size=1, fpath=str(params.job_dir / "model_flops.log"))

        model.summary(print_fn=logger.info)
        logger.info(f"Model requires {flops/1e6:0.2f} MFLOPS")

        logger.info("Performing inference")
        y_true = np.argmax(test_y, axis=2)
        y_prob = tf.nn.softmax(model.predict(test_x)).numpy()
        y_pred = np.argmax(y_prob, axis=2)
    # END WITH

    # Summarize results
    logger.info("Testing Results")
    test_acc = np.sum(y_pred == y_true) / y_true.size
    test_iou = compute_iou(y_true, y_pred, average="weighted")
    logger.info(f"[TEST SET] ACC={test_acc:.2%}, IoU={test_iou:.2%}")

    cm_path = str(params.job_dir / "confusion_matrix_test.png")
    class_names = get_class_names(HeartTask.segmentation)
    confusion_matrix_plot(
        y_true.flatten(),
        y_pred.flatten(),
        labels=class_names,
        save_path=cm_path,
        normalize="true",
    )


def export_model(params: HeartExportParams):
    """Export segmentation model.

    Args:
        params (HeartDemoParams): Deployment parameters
    """
    tfl_model_path = str(params.job_dir / "model.tflite")
    tflm_model_path = str(params.job_dir / "model_buffer.h")

    # Load model and set fixed batch size of 1
    logger.info("Loading trained model")
    with tfmot.quantization.keras.quantize_scope():
        model = load_model(str(params.model_file))

    in_shape, _ = get_task_shape(HeartTask.segmentation, params.frame_size)
    inputs = tf.keras.layers.Input(in_shape, dtype=tf.float32, batch_size=1)
    outputs = model(inputs)
    if not params.use_logits and not isinstance(model.layers[-1], tf.keras.layers.Softmax):
        outputs = tf.keras.layers.Softmax()(outputs)
        model = tf.keras.Model(inputs, outputs, name=model.name)
        outputs = model(inputs)
    # END IF
    flops = get_flops(model, batch_size=1, fpath=str(params.job_dir / "model_flops.log"))
    model.summary(print_fn=logger.info)

    logger.info(f"Model requires {flops/1e6:0.2f} MFLOPS")

    test_x, test_y = load_test_datasets(params)

    logger.info("Converting model to TFLite")
    tflite_model = convert_tflite(
        model=model,
        quantize=params.quantization,
        test_x=test_x,
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
    y_true = np.argmax(test_y, axis=2)
    y_pred_tf = np.argmax(model.predict(test_x), axis=2)
    y_pred_tfl = np.argmax(predict_tflite(model_content=tflite_model, test_x=test_x), axis=2)

    tf_acc = np.sum(y_true == y_pred_tf) / y_true.size
    tf_iou = compute_iou(y_true, y_pred_tf, average="weighted")
    logger.info(f"[TF SET] ACC={tf_acc:.2%}, IoU={tf_iou:.2%}")

    tfl_acc = np.sum(y_true == y_pred_tfl) / y_true.size
    tfl_iou = compute_iou(y_true, y_pred_tfl, average="weighted")
    logger.info(f"[TFL SET] ACC={tfl_acc:.2%}, IoU={tfl_iou:.2%}")

    # Check accuracy hit
    tfl_acc_drop = max(0, tf_acc - tfl_acc)
    if params.val_acc_threshold is not None and (1 - tfl_acc_drop) < params.val_acc_threshold:
        logger.warning(f"TFLite accuracy dropped by {tfl_acc_drop:0.2%}")
    elif params.val_acc_threshold:
        logger.info(f"Validation passed ({tfl_acc_drop:0.2%})")

    if params.tflm_file and tflm_model_path != params.tflm_file:
        logger.info(f"Copying TFLM header to {params.tflm_file}")
        shutil.copyfile(tflm_model_path, params.tflm_file)
