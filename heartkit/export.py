import shutil

import numpy as np
import pydantic_argparse
import tensorflow as tf
from rich.console import Console
from sklearn.metrics import f1_score

from neuralspot.tflite.convert import convert_tflite, predict_tflite, xxd_c_dump
from neuralspot.tflite.model import load_model

from .datasets.icentia11k import IcentiaDataset
from .models.utils import get_predicted_threshold_indices
from .types import HeartExportParams
from .utils import setup_logger

console = Console()

logger = setup_logger(__name__)


def export_model(params: HeartExportParams):
    """Deploy model command. This will convert saved model to TFLite and TFLite micro.
    Args:
        params (HeartDemoParams): Deployment parameters
    """
    tfl_model_path = str(params.job_dir / "model.tflite")
    tflm_model_path = str(params.job_dir / "model_buffer.h")

    # Load model and set fixed batch size of 1
    logger.info("Loading trained model")
    model = load_model(str(params.model_file))
    input_layer = tf.keras.layers.Input(
        (params.frame_size, 1), dtype=tf.float32, batch_size=1
    )
    model(input_layer)

    # Load dataset
    with console.status("[bold green] Loading test dataset..."):
        ds = IcentiaDataset(
            ds_path=str(params.ds_path),
            task=params.task,
            frame_size=params.frame_size,
        )
        test_ds = ds.load_test_dataset(
            # test_patients=params.test_patients,
            test_pt_samples=params.samples_per_patient,
            # num_workers=params.data_parallelism,
        )
        test_x, test_y = next(test_ds.batch(100000).as_numpy_iterator())
    # END WITH

    logger.info("Converting model to TFLite")
    tflite_model = convert_tflite(
        model,
        quantize=params.quantization,
        test_x=test_x[:1000],
        input_type=tf.int8 if params.quantization else None,
        output_type=tf.int8 if params.quantization else None,
    )

    # Save TFLite model
    logger.info(f"Saving TFLite model to {tfl_model_path}")
    with open(tfl_model_path, "wb") as fp:
        fp.write(tflite_model)

    # Save TFLM model
    logger.info(f"Saving TFL micro model to {tflm_model_path}")
    xxd_c_dump(
        src_path=tfl_model_path,
        dst_path=tflm_model_path,
        var_name=params.tflm_var_name,
        chunk_len=12,
        is_header=True,
    )

    # Verify TFLite results match TF results on example data
    logger.info("Validating model results")
    y_true = test_y
    y_prob_tf = tf.nn.softmax(model.predict(test_x)).numpy()
    y_pred_tf = np.argmax(y_prob_tf, axis=1)
    y_prob_tfl = tf.nn.softmax(
        predict_tflite(model_content=tflite_model, test_x=test_x)
    ).numpy()
    y_pred_tfl = np.argmax(y_prob_tfl, axis=1)

    tf_acc = np.sum(y_pred_tf == y_true) / len(y_true)
    tf_f1 = f1_score(y_true, y_pred_tf, average="macro")
    tfl_acc = np.sum(y_pred_tfl == y_true) / len(y_true)
    tfl_f1 = f1_score(y_true, y_pred_tfl, average="macro")
    logger.info(f"[TEST SET]  TF: ACC={tf_acc:.2%}, F1={tf_f1:.2%}")
    logger.info(f"[TEST SET] TFL: ACC={tfl_acc:.2%}, F1={tfl_f1:.2%}")

    if params.threshold is not None:
        y_thresh_idx = np.union1d(
            get_predicted_threshold_indices(y_prob_tf, y_pred_tf, params.threshold),
            get_predicted_threshold_indices(y_prob_tfl, y_pred_tfl, params.threshold),
        )
        y_thresh_idx.sort()
        drop_perc = 1 - len(y_thresh_idx) / len(y_true)
        y_pred_tf = y_pred_tf[y_thresh_idx]
        y_pred_tfl = y_pred_tfl[y_thresh_idx]
        y_true = y_true[y_thresh_idx]

        tf_acc = np.sum(y_pred_tf == y_true) / len(y_true)
        tf_f1 = f1_score(y_true, y_pred_tf, average="macro")
        tfl_acc = np.sum(y_pred_tfl == y_true) / len(y_true)
        tfl_f1 = f1_score(y_true, y_pred_tfl, average="macro")

        logger.info(
            f"[TEST SET]  TF: ACC={tf_acc:.2%}, F1={tf_f1:.2%}, THRESH={params.threshold:0.2%}, DROP={drop_perc:.2%}"
        )
        logger.info(
            f"[TEST SET] TFL: ACC={tfl_acc:.2%}, F1={tfl_f1:.2%}, THRESH={params.threshold:0.2%}, DROP={drop_perc:.2%}"
        )
    # END IF

    # Check accuracy hit
    acc_diff = tf_acc - tfl_acc
    if acc_diff > 0.5:
        logger.warning(f"TFLite accuracy dropped by {100*acc_diff:0.2f}%")
    else:
        logger.info("Validation passed")

    if params.tflm_file and tflm_model_path != params.tflm_file:
        logger.info(f"Copying TFLM header to {params.tflm_file}")
        shutil.copyfile(tflm_model_path, params.tflm_file)


def create_parser():
    """Create CLI argument parser
    Returns:
        ArgumentParser: Arg parser
    """
    return pydantic_argparse.ArgumentParser(
        model=HeartExportParams,
        prog="Heart deploy command",
        description="Deploy heart model to EVB",
    )


if __name__ == "__main__":
    parser = create_parser()
    export_model(parser.parse_typed_args())
