import logging
import os

import numpy as np
import tensorflow as tf

import neuralspot_edge as nse
from ...defines import HKTestParams
from ...metrics import compute_iou
from ...utils import set_random_seed, setup_logger
from ..utils import load_datasets
from .datasets import load_test_dataset


def evaluate(params: HKTestParams):
    """Evaluate model

    Args:
        params (HKTestParams): Evaluation parameters
    """
    logger = setup_logger(__name__, level=params.verbose)

    params.seed = set_random_seed(params.seed)
    logger.debug(f"Random seed {params.seed}")

    os.makedirs(params.job_dir, exist_ok=True)
    logger.debug(f"Creating working directory in {params.job_dir}")

    handler = logging.FileHandler(params.job_dir / "test.log", mode="w")
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)

    class_names = params.class_names or [f"Class {i}" for i in range(params.num_classes)]

    feat_shape = (params.frame_size, 1)
    class_shape = (params.frame_size, params.num_classes)

    ds_spec = (
        tf.TensorSpec(shape=feat_shape, dtype="float32"),
        tf.TensorSpec(shape=class_shape, dtype="int32"),
    )

    datasets = load_datasets(datasets=params.datasets)

    test_ds = load_test_dataset(datasets=datasets, params=params, ds_spec=ds_spec)
    test_x, test_y = next(test_ds.batch(params.test_size).as_numpy_iterator())

    logger.debug("Loading model")
    model = nse.models.load_model(params.model_file)
    flops = nse.metrics.flops.get_flops(model, batch_size=1, fpath=params.job_dir / "model_flops.log")

    model.summary(print_fn=logger.debug)
    logger.debug(f"Model requires {flops/1e6:0.2f} MFLOPS")

    logger.debug("Performing inference")
    y_true = np.argmax(test_y, axis=-1)
    y_pred = np.argmax(model.predict(test_x), axis=-1)

    # Summarize results
    logger.info("Testing Results")
    test_acc = np.sum(y_pred == y_true) / y_true.size
    test_iou = compute_iou(y_true, y_pred, average="weighted")
    logger.info(f"[TEST SET] ACC={test_acc:.2%}, IoU={test_iou:.2%}")
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    cm_path = params.job_dir / "confusion_matrix_test.png"
    nse.plotting.cm.confusion_matrix_plot(y_true, y_pred, labels=class_names, save_path=cm_path, normalize="true")
    nse.plotting.cm.px_plot_confusion_matrix(
        y_true,
        y_pred,
        labels=class_names,
        save_path=cm_path.with_suffix(".html"),
        normalize="true",
    )
