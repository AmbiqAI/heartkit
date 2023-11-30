import logging
import os

import numpy as np
import tensorflow as tf
import tensorflow_model_optimization as tfmot

from .. import tflite as tfa
from ..defines import HeartTestParams
from ..metrics import compute_iou, confusion_matrix_plot
from ..utils import set_random_seed, setup_logger
from .defines import get_class_mapping, get_class_names
from .utils import load_datasets, load_test_datasets

logger = setup_logger(__name__)


def evaluate(params: HeartTestParams):
    """Test segmentation model.

    Args:
        params (HeartTestParams): Testing/evaluation parameters
    """

    params.datasets = getattr(params, "datasets", ["ludb"])
    params.num_pts = getattr(params, "num_pts", 1000)
    params.seed = set_random_seed(params.seed)

    os.makedirs(params.job_dir, exist_ok=True)
    logger.info(f"Creating working directory in {params.job_dir}")

    handler = logging.FileHandler(params.job_dir / "test.log", mode="w")
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)

    logger.info(f"Random seed {params.seed}")

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
    test_x, test_y = load_test_datasets(datasets, params)

    with tfmot.quantization.keras.quantize_scope():
        logger.info("Loading model")
        model = tfa.load_model(params.model_file)
        flops = tfa.get_flops(model, batch_size=1, fpath=params.job_dir / "model_flops.log")

        model.summary(print_fn=logger.info)
        logger.info(f"Model requires {flops/1e6:0.2f} MFLOPS")

        logger.info("Performing inference")
        y_true = np.argmax(test_y, axis=-1)
        y_prob = tf.nn.softmax(model.predict(test_x)).numpy()
        y_pred = np.argmax(y_prob, axis=-1)
    # END WITH

    # Summarize results
    logger.info("Testing Results")
    test_acc = np.sum(y_pred == y_true) / y_true.size
    test_iou = compute_iou(y_true, y_pred, average="weighted")
    logger.info(f"[TEST SET] ACC={test_acc:.2%}, IoU={test_iou:.2%}")

    confusion_matrix_plot(
        y_true.flatten(),
        y_pred.flatten(),
        labels=class_names,
        save_path=params.job_dir / "confusion_matrix_test.png",
        normalize="true",
    )
