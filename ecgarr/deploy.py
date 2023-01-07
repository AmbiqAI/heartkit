import shutil
from typing import List, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt
import pydantic_argparse
import tensorflow as tf
from rich.console import Console
from sklearn.metrics import f1_score

from neuralspot.tflite.convert import convert_tflite, predict_tflite

from . import datasets as ds
from .models.utils import get_predicted_threshold_indices, load_model
from .types import EcgDeployParams, EcgTask
from .utils import setup_logger, xxd_c_dump

console = Console()

logger = setup_logger(__name__)


def create_dataset(
    db_path: str,
    task: EcgTask = EcgTask.rhythm,
    frame_size: Optional[int] = 1250,
    num_patients: int = 100,
    samples_per_patient: Union[int, List[int]] = 100,
    sample_size: Optional[int] = None,
    normalize: bool = True,
) -> Tuple[npt.ArrayLike, npt.ArrayLike]:
    """Generate test dataset

    Args:
        db_path (str): Database path
        task (EcgTask, optional): ECG Task. Defaults to EcgTask.rhythm.
        frame_size (Optional[int], optional): ECG Frame size. Defaults to 1250.
        num_patients (int, optional): # of patients. Defaults to 100.
        samples_per_patient (int, optional): # samples per patient. Defaults to 100.

    Returns:
        Tuple[npt.ArrayLike, npt.ArrayLike]: (test_x, test_y)
    """
    if sample_size is None:
        sample_size = 100 * num_patients
    patient_ids = ds.icentia11k.get_train_patient_ids()[:num_patients]
    np.random.shuffle(patient_ids)
    dataset = ds.create_dataset_from_generator(
        task=task,
        db_path=db_path,
        patient_ids=patient_ids,
        frame_size=frame_size,
        samples_per_patient=samples_per_patient,
        normalize=normalize,
        repeat=False,
    )

    data_x, data_y = next(dataset.batch(sample_size).as_numpy_iterator())
    return data_x, data_y


def deploy_model(params: EcgDeployParams):
    """Deploy model command. This will convert saved model to TFLite and TFLite micro.
    Args:
        params (EcgDeployParams): Deployment parameters
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
    with console.status("[bold green] Loading dataset..."):
        test_x, test_y = create_dataset(
            db_path=str(params.db_path),
            task=params.task,
            frame_size=params.frame_size,
            num_patients=1000,
            samples_per_patient=params.samples_per_patient,
            sample_size=100000,
        )
    # END WITH

    logger.info("Converting model to TFLite")
    tflite_model = convert_tflite(
        model,
        quantize=params.quantization,
        test_x=test_x[:1000],
        # NOTE: Make input/output uint8
        input_type=tf.float32,
        output_type=tf.float32,
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

    if params.tflm_file:
        logger.info(f"Copying TFLM header to {params.tflm_file}")
        shutil.copyfile(tflm_model_path, params.tflm_file)


def create_parser():
    """Create CLI argument parser
    Returns:
        ArgumentParser: Arg parser
    """
    return pydantic_argparse.ArgumentParser(
        model=EcgDeployParams,
        prog="Heart arrhythmia deploy command",
        description="Deploy heart arrhythmia model to EVB",
    )


if __name__ == "__main__":
    parser = create_parser()
    deploy_model(parser.parse_typed_args())
