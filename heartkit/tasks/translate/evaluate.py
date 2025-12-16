import os
import json

import keras
import tensorflow as tf
import helia_edge as helia
from ...defines import HKTaskParams

from ...datasets import DatasetFactory
from .datasets import load_test_dataset


def evaluate(params: HKTaskParams):
    """Evaluate translation task model with given parameters.

    Args:
        params (HKTaskParams): Task parameters
    """
    logger = helia.utils.setup_logger(__name__, level=params.verbose, file_path=params.job_dir / "test.log")

    params.seed = helia.utils.set_random_seed(params.seed)
    logger.debug(f"Random seed {params.seed}")

    os.makedirs(params.job_dir, exist_ok=True)
    logger.debug(f"Creating working directory in {params.job_dir}")

    datasets = [DatasetFactory.get(ds.name)(**ds.params) for ds in params.datasets]

    # Load validation data
    if params.val_file:
        logger.info(f"Loading validation dataset from {params.val_file}")
        test_ds = tf.data.Dataset.load(str(params.val_file))
    else:
        test_ds = load_test_dataset(datasets=datasets, params=params)

    test_x, test_y = next(test_ds.batch(params.test_size).as_numpy_iterator())

    logger.debug("Loading model")
    model = helia.models.load_model(params.model_file)
    flops = helia.metrics.flops.get_flops(model, batch_size=1, fpath=params.job_dir / "model_flops.log")

    model.summary(print_fn=logger.info)
    logger.debug(f"Model requires {flops / 1e6:0.2f} MFLOPS")

    logger.debug("Performing inference")
    # y_true = test_y.squeeze()
    # y_prob = model.predict(test_x)
    # y_pred = y_prob.squeeze()

    # Summarize results
    metrics = model.evaluate(test_x, test_y, verbose=params.verbose, return_dict=True)
    logger.info("[TEST SET] " + ", ".join([f"{k.upper()}={v:.4f}" for k, v in metrics.items()]))

    rst = {}
    rst["flops"] = flops
    rst["parameters"] = model.count_params()
    with open(params.job_dir / "metrics.json", "w") as fp:
        json.dump(rst, fp)

    # cleanup
    keras.utils.clear_session()
    for ds in datasets:
        ds.close()
