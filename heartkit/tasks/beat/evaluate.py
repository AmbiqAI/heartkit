import os
import json

import keras
import numpy as np
import tensorflow as tf
import neuralspot_edge as nse

from ...defines import HKTaskParams
from ...datasets import DatasetFactory
from .datasets import load_test_dataset


def evaluate(params: HKTaskParams):
    """Evaluate beat task model on given parameters.

    Args:
        params (HKTaskParams): Evaluation parameters
    """
    os.makedirs(params.job_dir, exist_ok=True)
    logger = nse.utils.setup_logger(__name__, level=params.verbose, file_path=params.job_dir / "test.log")
    logger.debug(f"Creating working directory in {params.job_dir}")

    params.seed = nse.utils.set_random_seed(params.seed)
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
    model = nse.models.load_model(params.model_file)
    flops = nse.metrics.flops.get_flops(model, batch_size=1, fpath=params.job_dir / "model_flops.log")

    model.summary(print_fn=logger.debug)
    logger.debug(f"Model requires {flops/1e6:0.2f} MFLOPS")

    logger.debug("Performing inference")
    y_true = np.argmax(test_y, axis=-1).flatten()
    y_prob = keras.ops.softmax(model.predict(test_x)).numpy()
    y_pred = np.argmax(y_prob, axis=-1).flatten()

    # Summarize results
    logger.debug("Testing Results")
    rst = model.evaluate(test_x, test_y, verbose=params.verbose, return_dict=True)
    logger.info("[TEST SET] " + ", ".join([f"{k.upper()}={v:.4f}" for k, v in rst.items()]))

    if params.num_classes == 2:
        roc_path = params.job_dir / "roc_auc_test.png"
        nse.plotting.roc_auc_plot(y_true, y_prob[:, 1], labels=class_names, save_path=roc_path)
    # END IF

    # If threshold given, only count predictions above threshold
    if params.threshold:
        prev_numel = len(y_true)
        indices = nse.metrics.threshold.get_predicted_threshold_indices(y_prob, y_pred, params.threshold)
        test_x, test_y = test_x[indices], test_y[indices]
        y_true, y_pred = y_true[indices], y_pred[indices]
        rst = model.evaluate(test_x, test_y, verbose=params.verbose, return_dict=True)
        logger.info(f"[TEST SET] THRESH={params.threshold:0.2%}, DROP={1 - len(indices) / prev_numel:.2%}")
        logger.info("[TEST SET] " + ", ".join([f"{k.upper()}={v:.4f}" for k, v in rst.items()]))
    # END IF

    cm_path = params.job_dir / "confusion_matrix_test.png"
    nse.plotting.confusion_matrix_plot(y_true, y_pred, labels=class_names, save_path=cm_path, normalize="true")
    nse.plotting.px_plot_confusion_matrix(
        y_true,
        y_pred,
        labels=class_names,
        save_path=cm_path.with_suffix(".html"),
        normalize="true",
    )

    rst["flops"] = flops
    rst["parameters"] = model.count_params()
    with open(params.job_dir / "metrics.json", "w") as fp:
        json.dump(rst, fp)

    # cleanup
    keras.utils.clear_session()
    for ds in datasets:
        ds.close()
