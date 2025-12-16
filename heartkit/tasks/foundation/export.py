import os
import shutil

import keras
import numpy as np
import tensorflow as tf
import helia_edge as helia

from ...defines import HKTaskParams
from ...datasets import DatasetFactory
from .datasets import load_test_dataset


def export(params: HKTaskParams):
    """Export foundation model

    Args:
        params (HKTaskParams): Task parameters
    """
    os.makedirs(params.job_dir, exist_ok=True)
    logger = helia.utils.setup_logger(__name__, level=params.verbose, file_path=params.job_dir / "export.log")
    logger.debug(f"Creating working directory in {params.job_dir}")

    tfl_model_path = params.job_dir / "model.tflite"
    tflm_model_path = params.job_dir / "model_buffer.h"

    feat_shape = (params.frame_size, 1)

    datasets = [DatasetFactory.get(ds.name)(**ds.params) for ds in params.datasets]

    # Load validation data
    if params.val_file:
        logger.info(f"Loading validation dataset from {params.val_file}")
        test_ds = tf.data.Dataset.load(str(params.val_file))
    else:
        test_ds = load_test_dataset(datasets=datasets, params=params)

    test_x = np.concatenate([x[helia.trainers.SimCLRTrainer.SAMPLES] for x in test_ds.as_numpy_iterator()])

    # Load model and set fixed batch size of 1
    logger.debug("Loading trained model")
    model = helia.models.load_model(params.model_file)

    inputs = keras.Input(shape=feat_shape, batch_size=1, dtype="float32")
    model(inputs)

    flops = helia.metrics.flops.get_flops(model, batch_size=1, fpath=params.job_dir / "model_flops.log")
    model.summary(print_fn=logger.info)
    logger.debug(f"Model requires {flops / 1e6:0.2f} MFLOPS")

    logger.debug(f"Converting model to TFLite (quantization={params.quantization.mode})")
    converter = helia.converters.tflite.TfLiteKerasConverter(model=model)

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

    tflite = helia.interpreters.tflite.TfLiteKerasInterpreter(tflite_content)
    tflite.compile()

    logger.debug("Validating model results")
    y_pred_tf = model.predict(test_x)
    y_pred_tfl = tflite.predict(x=test_x)

    metrics = [
        keras.metrics.CosineSimilarity(name="cos"),
        keras.metrics.MeanSquaredError(name="mse"),
    ]

    tfl_rst = helia.metrics.compute_metrics(metrics, y_pred_tf, y_pred_tfl)
    logger.info("[TFL METRICS] " + " ".join([f"{k.upper()}={v:.4f}" for k, v in tfl_rst.items()]))

    if params.tflm_file and tflm_model_path != params.tflm_file:
        logger.debug(f"Copying TFLM header to {params.tflm_file}")
        shutil.copyfile(tflm_model_path, params.tflm_file)

    # cleanup
    keras.utils.clear_session()
    for ds in datasets:
        ds.close()
