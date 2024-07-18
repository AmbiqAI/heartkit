import logging
import os

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report, f1_score

import neuralspot_edge as nse
from ...defines import HKTestParams
from ...utils import set_random_seed, setup_logger
from ..utils import load_datasets
from .datasets import load_test_dataset

logger = setup_logger(__name__)


def evaluate(params: HKTestParams):
    """Evaluate model

    Args:
        params (HKTestParams): Evaluation parameters
    """
    params.threshold = params.threshold or 0.5

    params.seed = set_random_seed(params.seed)
    logger.debug(f"Random seed {params.seed}")

    os.makedirs(params.job_dir, exist_ok=True)
    logger.debug(f"Creating working directory in {params.job_dir}")

    handler = logging.FileHandler(params.job_dir / "test.log", mode="w")
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)

    # classes = sorted(list(set(params.class_map.values())))
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

    model.summary(print_fn=logger.info)
    logger.debug(f"Model requires {flops/1e6:0.2f} MFLOPS")

    logger.debug("Performing inference")
    y_true = test_y
    y_prob = model.predict(test_x)
    y_pred = y_prob >= params.threshold

    cm_path = params.job_dir / "confusion_matrix_test.png"
    nse.plotting.cm.multilabel_confusion_matrix_plot(
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
    df_report.to_csv(params.job_dir / "classification_report_test.csv")
    test_acc = np.sum(y_pred == y_true) / y_true.size
    test_f1 = f1_score(y_true, y_pred, average="weighted")
    logger.info(f"[TEST SET] ACC={test_acc:.2%}, F1={test_f1:.2%}")
