import logging
import os

import numpy as np
import tensorflow as tf
from sklearn.metrics import f1_score

import neuralspot_edge as nse
from ...defines import HKTestParams
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
    class_shape = (params.num_classes,)

    ds_spec = (
        tf.TensorSpec(shape=feat_shape, dtype=tf.float32),
        tf.TensorSpec(shape=class_shape, dtype=tf.int32),
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
    y_prob = tf.nn.softmax(model.predict(test_x)).numpy()
    y_pred = np.argmax(y_prob, axis=-1)

    # Summarize results
    logger.info("Testing Results")
    test_acc = np.sum(y_pred == y_true) / len(y_true)
    test_f1 = f1_score(y_true, y_pred, average="weighted")
    logger.info(f"[TEST SET] ACC={test_acc:.2%}, F1={test_f1:.2%}")

    if params.num_classes == 2:
        roc_path = params.job_dir / "roc_auc_test.png"
        nse.plotting.roc.roc_auc_plot(y_true, y_prob[:, 1], labels=class_names, save_path=roc_path)
    # END IF

    # If threshold given, only count predictions above threshold
    if params.threshold:
        prev_numel = len(y_true)
        y_prob, y_pred, y_true = nse.metrics.threshold.threshold_predictions(y_prob, y_pred, y_true, params.threshold)
        drop_perc = 1 - len(y_true) / prev_numel
        test_acc = np.sum(y_pred == y_true) / len(y_true)
        test_f1 = f1_score(y_true, y_pred, average="weighted")
        logger.info(f"[TEST SET] THRESH={params.threshold:0.2%}, DROP={drop_perc:.2%}")
        logger.info(f"[TEST SET] ACC={test_acc:.2%}, F1={test_f1:.2%}")
    # END IF

    cm_path = params.job_dir / "confusion_matrix_test.png"
    nse.plotting.cm.confusion_matrix_plot(y_true, y_pred, labels=class_names, save_path=cm_path, normalize="true")
    nse.plotting.cm.px_plot_confusion_matrix(
        y_true,
        y_pred,
        labels=class_names,
        save_path=cm_path.with_suffix(".html"),
        normalize="true",
    )
