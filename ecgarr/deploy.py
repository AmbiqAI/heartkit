import functools
import logging
from typing import Optional
import numpy as np
import tensorflow as tf
import pydantic_argparse
from .utils import xxd_c_dump, setup_logger
from . import datasets as ds
from .types import EcgTask, EcgDeployParams

logger = logging.getLogger('ecgarr.deploy')

def create_dataset(
        db_path: str,
        task: EcgTask = EcgTask.rhythm,
        frame_size: Optional[int] = 1250,
        num_patients: int = 100,
        samples_per_patient: int = 100,
    ):
    # TODO: Infer frame_size from model
    sample_size = num_patients*samples_per_patient

    patient_ids = ds.icentia11k.get_train_patient_ids()[:num_patients]
    np.random.shuffle(patient_ids)
    dataset = ds.create_dataset_from_generator(
        task=task, db_path=db_path,
        patient_ids=patient_ids, frame_size=frame_size,
        samples_per_patient=samples_per_patient, repeat=False
    )
    data_x, _ = next(dataset.batch(sample_size).as_numpy_iterator())
    for i in data_x.shape:
        yield([data_x[i]])

def deploy_model(params: EcgDeployParams):
    setup_logger('ecgarr', str(params.job_dir))

    tfl_model_path = str(params.job_dir / 'model.tflite')
    tflm_model_path = str(params.job_dir / 'model.cc')

    # First load the model and provide fixed batch size
    model = tf.keras.models.load_model(params.model_file)
    input_layer = tf.keras.layers.Input((params.frame_size, 1), dtype=tf.float32, batch_size=1)
    output_layer = model(input_layer)

    # Instantiate converter from model
    converter = tf.lite.TFLiteConverter.from_keras_model(model=model)

    if params.quantization:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.uint8
        converter.inference_output_type = tf.uint8
        converter.representative_dataset = functools.partial(create_dataset,
            db_path=str(params.db_path),
            task=params.task,
            frame_size=params.frame_size,
            num_patients=100,
            samples_per_patient=100
        )
    # Convert model
    model_tflite = converter.convert()

    # Save TFLite model
    with open(tfl_model_path, 'wb') as fp:
        fp.write(model_tflite)

    # Save TF Micro model
    xxd_c_dump(tfl_model_path, tflm_model_path, var_name=params.tflm_var_name, chunk_len=12)


def create_parser():
    return pydantic_argparse.ArgumentParser(
        model=EcgDeployParams,
        prog="ECG Arrhythmia Deploy",
        description="ECG Arrhythmia Deployment"
    )

if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_typed_args()
    deploy_model(args)
