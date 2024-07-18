import logging
import os

import keras
import numpy as np
import tensorflow as tf

import neuralspot_edge as nse
from ...defines import HKExportParams
from ...utils import setup_logger
from ..utils import load_datasets
from .datasets import load_test_dataset

logger = setup_logger(__name__)


def export(params: HKExportParams):
    """Export model

    Args:
        params (HKExportParams): Deployment parameters
    """

    os.makedirs(params.job_dir, exist_ok=True)
    logger.debug(f"Creating working directory in {params.job_dir}")

    handler = logging.FileHandler(params.job_dir / "export.log", mode="w")
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)

    feat_shape = (params.frame_size, 1)

    tfl_model_path = params.job_dir / "model.tflite"
    tflm_model_path = params.job_dir / "model_buffer.h"

    ds_spec = (
        tf.TensorSpec(shape=feat_shape, dtype="float32"),
        tf.TensorSpec(shape=feat_shape, dtype="float32"),
    )

    datasets = load_datasets(datasets=params.datasets)

    test_ds = load_test_dataset(datasets=datasets, params=params, ds_spec=ds_spec)
    test_x, _ = next(test_ds.batch(params.test_size).as_numpy_iterator())

    # Load model and set fixed batch size of 1
    logger.debug("Loading trained model")
    model = nse.models.load_model(params.model_file)

    inputs = keras.Input(shape=ds_spec[0].shape, batch_size=1, dtype=ds_spec[0].dtype)
    model(inputs)

    flops = nse.metrics.flops.get_flops(model, batch_size=1, fpath=params.job_dir / "model_flops.log")
    model.summary(print_fn=logger.info)
    logger.debug(f"Model requires {flops/1e6:0.2f} MFLOPS")

    logger.debug(f"Converting model to TFLite (quantization={params.quantization.mode})")
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

    # Verify TFLite results match TF results
    logger.debug("Validating model results")
    y_pred_tf = model.predict(test_x)
    y_pred_tfl = tflite.predict(x=test_x)

    # Compare error between TF and TFLite outputs
    error = np.abs(y_pred_tf - y_pred_tfl).max()
    logger.info(f"Max error between TF and TFLite outputs: {error}")
