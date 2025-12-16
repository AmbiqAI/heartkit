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
    """Export model for denoise task with given parameters.

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

    test_x = np.concatenate([x for x, _ in test_ds.as_numpy_iterator()])
    test_y = np.concatenate([y for _, y in test_ds.as_numpy_iterator()])

    # Load model and set fixed batch size of 1
    logger.debug("Loading trained model")
    model = helia.models.load_model(params.model_file)

    # Add softmax layer if required
    if getattr(params, "flatten", False):
        model = helia.models.append_layers(model, layers=[keras.layers.Flatten()], copy_weights=True)
    # END IF

    inputs = keras.Input(shape=feat_shape, batch_size=1, name="input", dtype="float32")
    model(inputs)

    flops = helia.metrics.flops.get_flops(model, batch_size=1, fpath=params.job_dir / "model_flops.log")
    model.summary(print_fn=logger.debug)
    logger.debug(f"Model requires {flops / 1e6:0.2f} MFLOPS")

    logger.debug(f"Converting model to TFLite (quantization={params.quantization.format})")
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

    # Verify TFLite results match TF results on example data
    metrics = [
        keras.metrics.MeanSquaredError(name="loss"),
        keras.metrics.MeanAbsoluteError(name="mae"),
        keras.metrics.MeanSquaredError(name="mse"),
        keras.metrics.RootMeanSquaredError(name="rmse"),
        keras.metrics.CosineSimilarity(name="cosine"),
    ]

    if params.val_metric not in [m.name for m in metrics]:
        raise ValueError(f"Metric {params.val_metric} not supported")

    logger.info("Validating model results")
    y_true = test_y
    if getattr(params, "flatten", False):
        y_true = y_true.squeeze(axis=-1)
    y_pred_tf = model.predict(test_x)
    y_pred_tfl = tflite.predict(x=test_x)
    print(y_true.shape, y_pred_tf.shape, y_pred_tfl.shape)

    tf_rst = helia.metrics.compute_metrics(metrics, y_true, y_pred_tf)
    tfl_rst = helia.metrics.compute_metrics(metrics, y_true, y_pred_tfl)
    logger.info("[TF METRICS] " + " ".join([f"{k.upper()}={v:.4f}" for k, v in tf_rst.items()]))
    logger.info("[TFL METRICS] " + " ".join([f"{k.upper()}={v:.4f}" for k, v in tfl_rst.items()]))

    metric_diff = abs(tf_rst[params.val_metric] - tfl_rst[params.val_metric])

    # Check accuracy hit
    if params.test_metric_threshold is not None and metric_diff > params.test_metric_threshold:
        logger.warning(f"TFLite accuracy dropped by {metric_diff:0.4f}")
    elif params.test_metric_threshold:
        logger.info(f"Validation passed ({metric_diff:0.4f})")

    if params.tflm_file and tflm_model_path != params.tflm_file:
        logger.debug(f"Copying TFLM header to {params.tflm_file}")
        shutil.copyfile(tflm_model_path, params.tflm_file)

    # cleanup
    keras.utils.clear_session()
    for ds in datasets:
        ds.close()
