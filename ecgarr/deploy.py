import logging
from typing import Optional, Tuple
import numpy as np
import numpy.typing as npt
import tensorflow as tf
import pydantic_argparse
from rich.console import Console
from .utils import xxd_c_dump, setup_logger
from . import datasets as ds
from .types import EcgTask, EcgDeployParams

console = Console()
logger = logging.getLogger("ECGARR")


def create_dataset(
    db_path: str,
    task: EcgTask = EcgTask.rhythm,
    frame_size: Optional[int] = 1250,
    num_patients: int = 100,
    samples_per_patient: int = 100,
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

    sample_size = num_patients * samples_per_patient
    patient_ids = ds.icentia11k.get_train_patient_ids()[:num_patients]
    np.random.shuffle(patient_ids)
    dataset = ds.create_dataset_from_generator(
        task=task,
        db_path=db_path,
        patient_ids=patient_ids,
        frame_size=frame_size,
        samples_per_patient=samples_per_patient,
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
    tflm_model_path = str(params.job_dir / "model.h")

    # Load model and set fixed batch size of 1
    logger.info("Loading trained model")
    model = tf.keras.models.load_model(params.model_file)
    input_layer = tf.keras.layers.Input(
        (params.frame_size, 1), dtype=tf.float32, batch_size=1
    )
    model(input_layer)

    # Load dataset
    with console.status("[bold green] Loading dataset..."):
        test_x, _ = create_dataset(
            db_path=str(params.db_path),
            task=params.task,
            frame_size=params.frame_size,
            num_patients=200,
            samples_per_patient=10,
        )

    # Instantiate converter from model
    converter = tf.lite.TFLiteConverter.from_keras_model(model=model)

    # Optionally quantize model
    if params.quantization:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        # NOTE: Enable once QAT is working
        # converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        # converter.inference_input_type = tf.uint8
        # converter.inference_output_type = tf.uint8
        def rep_dataset():
            for i in range(test_x.shape[0]):
                yield [test_x[i : i + 1]]

        converter.representative_dataset = rep_dataset

    # Convert model
    logger.info("Converting model to TFLite")
    model_tflite = converter.convert()

    # Save TFLite model
    logger.info(f"Saving TFLite model to {tfl_model_path}")
    with open(tfl_model_path, "wb") as fp:
        fp.write(model_tflite)

    # Save TF Micro model
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
    interpreter = tf.lite.Interpreter(tfl_model_path)
    model_sig = interpreter.get_signature_runner()
    input_details = model_sig.get_input_details()
    output_details = model_sig.get_output_details()
    input_name = list(input_details.keys())[0]
    output_name = list(output_details.keys())[0]

    # Predict using TF
    y_prob_tf = tf.nn.softmax(model.predict(test_x)).numpy()
    y_pred_tf = np.argmax(y_prob_tf, axis=1)

    # Predict using TFLite
    y_prob_tfl = tf.nn.softmax(
        np.array(
            [
                model_sig(**{input_name: test_x[i : i + 1]})[output_name][0]
                for i in range(test_x.shape[0])
            ]
        )
    ).numpy()
    y_pred_tfl = np.argmax(y_prob_tfl, axis=1)

    # If threshold given, only count predictions above threshold
    # if params.threshold is not None:
    #     logger.info(f"Using threshold {params.threshold:0.2f}")
    #     y_pred_prob_tf = np.take_along_axis(
    #         y_prob_tf, np.expand_dims(y_pred_tf, axis=-1), axis=-1
    #     ).squeeze(axis=-1)

    #     y_pred_prob_tfl = np.take_along_axis(
    #         y_prob_tfl, np.expand_dims(y_pred_tfl, axis=-1), axis=-1
    #     ).squeeze(axis=-1)

    #     y_thresh_idx = np.union1d(
    #         np.where(y_pred_prob_tf > params.threshold)[0],
    #         np.where(y_pred_prob_tfl > params.threshold)[0]
    #     )
    #     y_thresh_idx.sort()

    #     logger.info(
    #         f"Dropping {len(y_thresh_idx)} samples ({100*(1-len(y_thresh_idx)/len(y_true)):0.2f}%)"
    #     )
    #     y_pred_tf = y_pred_tf[y_thresh_idx]
    #     y_pred_tfl = y_pred_tfl[y_thresh_idx]

    # Verify TF matches TFLite
    num_bad = np.sum(np.abs(y_pred_tfl - y_pred_tf))
    if num_bad:
        logger.warning(f"Found {num_bad} labels mismatched betwee TF and TFLite")
    else:
        logger.info("Validation passed")

    # Generate header with samples


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
    setup_logger("ECGARR")
    parser = create_parser()
    deploy_model(parser.parse_typed_args())
