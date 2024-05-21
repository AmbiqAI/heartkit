import logging
import os

import numpy as np
import tensorflow as tf
import tensorflow_model_optimization as tfmot

from ... import tflite as tfa
from ...defines import HKTestParams
from ...metrics import compute_iou, confusion_matrix_plot, px_plot_confusion_matrix
from ...utils import set_random_seed, setup_logger
from ..utils import load_datasets
from .datasets import load_test_dataset

logger = setup_logger(__name__)


def evaluate(params: HKTestParams):
    """Evaluate model

    Args:
        params (HKTestParams): Evaluation parameters
    """
    params.seed = set_random_seed(params.seed)
    logger.info(f"Random seed {params.seed}")

    os.makedirs(params.job_dir, exist_ok=True)
    logger.info(f"Creating working directory in {params.job_dir}")

    handler = logging.FileHandler(params.job_dir / "test.log", mode="w")
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)

    class_names = params.class_names or [f"Class {i}" for i in range(params.num_classes)]

    feat_shape = (params.frame_size, 1)
    class_shape = (params.frame_size, params.num_classes)

    ds_spec = (
        tf.TensorSpec(shape=feat_shape, dtype=tf.float32),
        tf.TensorSpec(shape=class_shape, dtype=tf.int32),
    )

    datasets = load_datasets(datasets=params.datasets)

    test_ds = load_test_dataset(datasets=datasets, params=params, ds_spec=ds_spec)
    test_x, test_y = next(test_ds.batch(params.test_size).as_numpy_iterator())

    with tfmot.quantization.keras.quantize_scope():
        logger.info("Loading model")
        model = tfa.load_model(params.model_file)
        flops = tfa.get_flops(model, batch_size=1, fpath=params.job_dir / "model_flops.log")

        model.summary(print_fn=logger.info)
        logger.info(f"Model requires {flops/1e6:0.2f} MFLOPS")

        logger.info("Performing inference")
        y_true = np.argmax(test_y, axis=-1)
        y_pred = np.argmax(model.predict(test_x), axis=-1)
    # END WITH

    # Summarize results
    logger.info("Testing Results")
    test_acc = np.sum(y_pred == y_true) / y_true.size
    test_iou = compute_iou(y_true, y_pred, average="weighted")
    logger.info(f"[TEST SET] ACC={test_acc:.2%}, IoU={test_iou:.2%}")
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    cm_path = params.job_dir / "confusion_matrix_test.png"
    confusion_matrix_plot(y_true, y_pred, labels=class_names, save_path=cm_path, normalize="true")
    px_plot_confusion_matrix(
        y_true, y_pred, labels=class_names, save_path=cm_path.with_suffix(".html"), normalize="true"
    )
