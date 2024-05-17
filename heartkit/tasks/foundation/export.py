import logging
import os

import keras
import numpy as np
import tensorflow as tf
import tensorflow_model_optimization as tfmot

from ... import tflite as tfa
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
    logger.info(f"Creating working directory in {params.job_dir}")

    handler = logging.FileHandler(params.job_dir / "export.log", mode="w")
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)

    feat_shape = (params.frame_size, 1)

    tfl_model_path = params.job_dir / "model.tflite"
    tflm_model_path = params.job_dir / "model_buffer.h"

    ds_spec = (
        tf.TensorSpec(shape=feat_shape, dtype=tf.float32),
        tf.TensorSpec(shape=feat_shape, dtype=tf.float32),
    )

    datasets = load_datasets(datasets=params.datasets)

    test_ds = load_test_dataset(datasets=datasets, params=params, ds_spec=ds_spec)
    test_x, _ = next(test_ds.batch(params.test_size).as_numpy_iterator())

    # Load model and set fixed batch size of 1
    logger.info("Loading trained model")
    with tfmot.quantization.keras.quantize_scope():
        model = tfa.load_model(params.model_file)

    inputs = keras.Input(shape=ds_spec[0].shape, batch_size=1, dtype=ds_spec[0].dtype)
    model(inputs)

    flops = tfa.get_flops(model, batch_size=1, fpath=params.job_dir / "model_flops.log")
    model.summary(print_fn=logger.info)
    logger.info(f"Model requires {flops/1e6:0.2f} MFLOPS")

    converter = tfa.create_tflite_converter(
        model=model,
        quantize=params.quantization.enabled,
        test_x=test_x,
        input_type=params.quantization.input_type,
        output_type=params.quantization.output_type,
        supported_ops=params.quantization.supported_ops,
        use_concrete=True,
        feat_shape=ds_spec[0].shape,
    )
    tflite_model = converter.convert()

    # Save TFLite model
    logger.info(f"Saving TFLite model to {tfl_model_path}")
    with open(tfl_model_path, "wb") as fp:
        fp.write(tflite_model)

    # Save TFLM model
    logger.info(f"Saving TFL micro model to {tflm_model_path}")
    tfa.xxd_c_dump(
        src_path=tfl_model_path,
        dst_path=tflm_model_path,
        var_name=params.tflm_var_name,
        chunk_len=20,
        is_header=True,
    )

    y_pred_tf = model.predict(test_x)
    y_pred_tfl = tfa.predict_tflite(model_content=tflite_model, test_x=test_x)
    print(y_pred_tf.shape)

    # Compare error between TF and TFLite outputs
    error = np.abs(y_pred_tf - y_pred_tfl).max()
    logger.info(f"Max error between TF and TFLite outputs: {error}")
