import os
import json

import keras
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
import helia_edge as helia

from ...defines import HKTaskParams
from ...datasets import DatasetFactory
from .datasets import load_test_dataset


def evaluate(params: HKTaskParams):
    """Evaluate diagnostic task model with given parameters.

    Args:
        params (HKTaskParams): Task parameters
    """
    os.makedirs(params.job_dir, exist_ok=True)
    logger = helia.utils.setup_logger(__name__, level=params.verbose, file_path=params.job_dir / "test.log")
    logger.debug(f"Creating working directory in {params.job_dir}")

    params.threshold = params.threshold or 0.5

    params.seed = helia.utils.set_random_seed(params.seed)
    logger.debug(f"Random seed {params.seed}")

    class_names = params.class_names or [f"Class {i}" for i in range(params.num_classes)]

    datasets = [DatasetFactory.get(ds.name)(**ds.params) for ds in params.datasets]

    # Load validation data
    if params.val_file:
        logger.info(f"Loading validation dataset from {params.val_file}")
        test_ds = tf.data.Dataset.load(str(params.val_file))
    else:
        test_ds = load_test_dataset(datasets=datasets, params=params)

    test_x = np.concatenate([x for x, _ in test_ds.as_numpy_iterator()])
    test_y = np.concatenate([y for _, y in test_ds.as_numpy_iterator()])

    logger.debug("Loading model")
    model = helia.models.load_model(params.model_file)
    flops = helia.metrics.flops.get_flops(model, batch_size=1, fpath=params.job_dir / "model_flops.log")

    model.summary(print_fn=logger.info)
    logger.debug(f"Model requires {flops / 1e6:0.2f} MFLOPS")

    logger.debug("Performing inference")
    y_true = test_y
    y_prob = model.predict(test_x)

    # y_pred = y_prob >= params.threshold

    y_pred = np.argmax(y_prob, axis=-1)
    y_true = np.argmax(y_true, axis=-1)

    cm_path = params.job_dir / "confusion_matrix_test.png"
    helia.plotting.confusion_matrix_plot(
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

    rst = model.evaluate(test_ds, verbose=params.verbose, return_dict=True)
    logger.info("[TEST SET] " + ", ".join([f"{k.upper()}={v:.4f}" for k, v in rst.items()]))

    rst["flops"] = flops
    rst["parameters"] = model.count_params()
    with open(params.job_dir / "metrics.json", "w") as fp:
        json.dump(rst, fp)

    # cleanup
    keras.utils.clear_session()
    for ds in datasets:
        ds.close()
