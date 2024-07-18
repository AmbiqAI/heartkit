import logging
import os
import shutil

import keras
import numpy as np
import tensorflow as tf

import neuralspot_edge as nse
from ...defines import HKExportParams
from ...utils import setup_logger
from ..utils import load_datasets
from .datasets import load_test_dataset


def export(params: HKExportParams):
    """Export model

    Args:
        params (HKExportParams): Deployment parameters
    """
    logger = setup_logger(__name__, level=params.verbose)

    os.makedirs(params.job_dir, exist_ok=True)
    logger.debug(f"Creating working directory in {params.job_dir}")

    handler = logging.FileHandler(params.job_dir / "export.log", mode="w")
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)

    tfl_model_path = params.job_dir / "model.tflite"
    tflm_model_path = params.job_dir / "model_buffer.h"

    feat_shape = (params.frame_size, 1)
    class_shape = (params.frame_size, 1)

    ds_spec = (
        tf.TensorSpec(shape=feat_shape, dtype="float32"),
        tf.TensorSpec(shape=class_shape, dtype="float32"),
    )

    datasets = load_datasets(datasets=params.datasets)

    test_ds = load_test_dataset(datasets=datasets, params=params, ds_spec=ds_spec)
    test_x, test_y = next(test_ds.batch(params.test_size).as_numpy_iterator())

    # Load model and set fixed batch size of 1
    logger.debug("Loading trained model")
    model = nse.models.load_model(params.model_file)

    inputs = keras.Input(shape=ds_spec[0].shape, batch_size=1, name="input", dtype=ds_spec[0].dtype)
    model(inputs)  # Build model with fixed batch size of 1

    flops = nse.metrics.flops.get_flops(model, batch_size=1, fpath=params.job_dir / "model_flops.log")
    model.summary(print_fn=logger.debug)
    logger.debug(f"Model requires {flops/1e6:0.2f} MFLOPS")

    logger.debug(f"Converting model to TFLite (quantization={params.quantization.format})")
    converter = nse.converters.tflite.TfLiteKerasConverter(model=model)

    tflite_content = converter.convert(
        test_x=test_x,
        quantization=params.quantization.format,
        io_type=params.quantization.io_type,
        mode=params.quantization.conversion,
        strict=not params.quantization.fallback,
    )

    if params.quantization.debug:
        quant_df = converter.debug_quantization()
        quant_df.to_csv(params.job_dir / "quant.csv")

    # Save TFLite model
    logger.debug(f"Saving TFLite model to {tfl_model_path}")
    converter.export(tfl_model_path)

    # Save TFLM model
    logger.debug(f"Saving TFL micro model to {tflm_model_path}")
    converter.export_header(tflm_model_path, name=params.tflm_var_name)
    converter.cleanup()

    tflite = nse.interpreters.tflite.TfLiteKerasInterpreter(tflite_content)
    tflite.compile()

    # Verify TFLite results match TF results on example data
    logger.info("Validating model results")
    y_true = test_y
    y_pred_tf = model.predict(test_x)
    y_pred_tfl = tflite.predict(x=test_x)

    tf_mae = np.mean(np.abs(y_true - y_pred_tf))
    tf_rmse = np.sqrt(np.mean((y_true - y_pred_tf) ** 2))
    logger.info(f"[TF SET] MAE={tf_mae:.2%}, RMSE={tf_rmse:.2%}")

    tfl_mae = np.mean(np.abs(y_true - y_pred_tfl))
    tfl_rmse = np.sqrt(np.mean((y_true - y_pred_tfl) ** 2))
    logger.info(f"[TFL SET] MAE={tfl_mae:.2%}, RMSE={tfl_rmse:.2%}")

    # Check accuracy hit
    tfl_acc_drop = max(0, tf_mae - tfl_mae)
    if params.val_acc_threshold is not None and (1 - tfl_acc_drop) < params.val_acc_threshold:
        logger.warning(f"TFLite accuracy dropped by {tfl_acc_drop:0.2%}")
    elif params.val_acc_threshold:
        logger.info(f"Validation passed ({tfl_acc_drop:0.2%})")

    if params.tflm_file and tflm_model_path != params.tflm_file:
        logger.debug(f"Copying TFLM header to {params.tflm_file}")
        shutil.copyfile(tflm_model_path, params.tflm_file)
