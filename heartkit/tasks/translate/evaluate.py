import logging
import os

import keras
import numpy as np
import tensorflow as tf

import keras_edge as kedge
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

    params.seed = set_random_seed(params.seed)
    logger.info(f"Random seed {params.seed}")

    os.makedirs(params.job_dir, exist_ok=True)
    logger.info(f"Creating working directory in {params.job_dir}")

    handler = logging.FileHandler(params.job_dir / "test.log", mode="w")
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)

    feat_shape = (params.frame_size, 1)
    class_shape = (params.frame_size, 1)

    ds_spec = (
        tf.TensorSpec(shape=feat_shape, dtype="float32"),
        tf.TensorSpec(shape=class_shape, dtype="float32"),
    )

    datasets = load_datasets(datasets=params.datasets)

    test_ds = load_test_dataset(datasets=datasets, params=params, ds_spec=ds_spec)
    test_x, test_y = next(test_ds.batch(params.test_size).as_numpy_iterator())

    logger.info("Loading model")
    model = kedge.models.load_model(params.model_file)
    flops = kedge.metrics.flops.get_flops(model, batch_size=1, fpath=params.job_dir / "model_flops.log")

    model.summary(print_fn=logger.info)
    logger.info(f"Model requires {flops/1e6:0.2f} MFLOPS")

    logger.info("Performing inference")
    y_true = test_y.squeeze()
    y_prob = model.predict(test_x)
    y_pred = y_prob.squeeze()

    # Summarize results
    cossim = keras.metrics.CosineSimilarity()
    cossim.update_state(y_true, y_pred)  # pylint: disable=E1102
    test_cossim = cossim.result().numpy()  # pylint: disable=E1102
    logger.info("Testing Results")
    mae = keras.metrics.MeanAbsoluteError()
    mae.update_state(y_true, y_pred)  # pylint: disable=E1102
    test_mae = mae.result().numpy()  # pylint: disable=E1102
    mse = keras.metrics.MeanSquaredError()
    mse.update_state(y_true, y_pred)  # pylint: disable=E1102
    test_mse = mse.result().numpy()  # pylint: disable=E1102
    np.sqrt(np.mean(np.square(y_true - y_pred)))
    logger.info(f"[TEST SET] MAE={test_mae:.2%}, MSE={test_mse:.2%}, COSSIM={test_cossim:.2%}")
