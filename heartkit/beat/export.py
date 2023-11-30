import logging
import os
import shutil

import numpy as np
import tensorflow as tf

from .. import tflite as tfa
from ..defines import HeartExportParams
from ..metrics import f1_score
from ..utils import setup_logger
from .defines import get_class_mapping
from .utils import load_dataset, load_test_dataset

logger = setup_logger(__name__)


def export(params: HeartExportParams):
    """Export beat-level model.

    Args:
        params (HeartExportParams): Deployment parameters
    """

    os.makedirs(params.job_dir, exist_ok=True)
    logger.info(f"Creating working directory in {params.job_dir}")

    handler = logging.FileHandler(params.job_dir / "export.log", mode="w")
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)

    tfl_model_path = params.job_dir / "model.tflite"
    tflm_model_path = params.job_dir / "model_buffer.h"

    class_map = get_class_mapping(params.num_classes)

    ds = load_dataset(
        ds_path=params.ds_path, frame_size=params.frame_size, sampling_rate=params.sampling_rate, class_map=class_map
    )
    test_x, test_y = load_test_dataset(ds, params)

    # Load model and set fixed batch size of 1
    logger.info("Loading trained model")
    model = tfa.load_model(params.model_file)

    inputs = tf.keras.layers.Input(ds.feat_shape, dtype=tf.float32, batch_size=1)
    outputs = model(inputs)

    if not params.use_logits and not isinstance(model.layers[-1], tf.keras.layers.Softmax):
        outputs = tf.keras.layers.Softmax()(outputs)
        model = tf.keras.Model(inputs, outputs, name=model.name)
        outputs = model(inputs)
    # END IF

    logger.info("Converting model to TFLite")
    tflite_model = tfa.convert_tflite(
        model=model,
        quantize=params.quantization,
        test_x=test_x,
        input_type=tf.int8 if params.quantization else None,
        output_type=tf.int8 if params.quantization else None,
    )

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

    # Verify TFLite results match TF results on example data
    logger.info("Validating model results")
    y_true = np.argmax(test_y, axis=1)
    y_pred_tf = np.argmax(model.predict(test_x), axis=1)
    y_pred_tfl = np.argmax(tfa.predict_tflite(model_content=tflite_model, test_x=test_x), axis=1)

    tf_acc = np.sum(y_true == y_pred_tf) / y_true.size
    tf_f1 = f1_score(y_true, y_pred_tf, average="weighted")
    logger.info(f"[TF SET] ACC={tf_acc:.2%}, F1={tf_f1:.2%}")

    tfl_acc = np.sum(y_true == y_pred_tfl) / y_true.size
    tfl_f1 = f1_score(y_true, y_pred_tfl, average="weighted")
    logger.info(f"[TFL SET] ACC={tfl_acc:.2%}, F1={tfl_f1:.2%}")

    # Check accuracy hit
    tfl_acc_drop = max(0, tf_acc - tfl_acc)
    if params.val_acc_threshold is not None and (1 - tfl_acc_drop) < params.val_acc_threshold:
        logger.warning(f"TFLite accuracy dropped by {tfl_acc_drop:0.2%}")
    elif params.val_acc_threshold:
        logger.info(f"Validation passed ({tfl_acc_drop:0.2%})")

    if params.tflm_file and tflm_model_path != params.tflm_file:
        logger.info(f"Copying TFLM header to {params.tflm_file}")
        shutil.copyfile(tflm_model_path, params.tflm_file)
