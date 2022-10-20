from typing import Dict, List, Optional, Tuple, Union
import tensorflow as tf
import numpy as np
import numpy.typing as npt
from keras.engine.keras_tensor import KerasTensor
from .features import ecg_feature_extractor
from ..types import EcgTask, ArchitectureType

InputShape = Union[Tuple[int], List[Tuple[int]], Dict[str, Tuple[int]]]


def build_input_tensor_from_shape(
    shape: InputShape, dtype: tf.DType = None, ignore_batch_dim: bool = False
):
    """Build input tensor from shape which can be used to initialize the weights of a model.

    Args:
        shape (InputShape]): Input Shape
        dtype (tf.DType, optional): _description_. Defaults to None.
        ignore_batch_dim (bool, optional): Ignore first dimension as batch. Defaults to False.

    Returns:
        tf.keras.layers.Input: Input layer
    """
    if isinstance(shape, (list, tuple)):
        return [
            build_input_tensor_from_shape(
                shape=shape[i],
                dtype=dtype[i] if dtype else None,
                ignore_batch_dim=ignore_batch_dim,
            )
            for i in range(len(shape))
        ]

    if isinstance(shape, dict):
        return {
            k: build_input_tensor_from_shape(
                shape=shape[k],
                dtype=dtype[k] if dtype else None,
                ignore_batch_dim=ignore_batch_dim,
            )
            for k in shape
        }

    if ignore_batch_dim:
        shape = shape[1:]
    return tf.keras.layers.Input(shape, dtype=dtype)


def generate_task_model(
    inputs: KerasTensor,
    task: EcgTask,
    arch: ArchitectureType = "resnet18",
    stages: Optional[int] = None,
) -> tf.keras.Model:
    """Generate model for given arrhythmia task

    Args:
        inputs (KerasTensor): Model inputs
        task (EcgTask): Heart arrhythmia task
        arch (ArchitectureType, optional): Architecture type. Defaults to 'resnet18'.
        stages (Optional[int], optional): # stages in network. Defaults to None.

    Returns:
        tf.keras.Model: Model
    """

    if task == EcgTask.rhythm:
        num_classes = 2
    elif task == EcgTask.beat:
        num_classes = 5
    elif task == EcgTask.hr:
        num_classes = 4
    else:
        raise ValueError("unknown task: {}".format(task))
    x = ecg_feature_extractor(inputs, arch, stages=stages)
    outputs = tf.keras.layers.Dense(num_classes)(x)
    model = tf.keras.Model(inputs, outputs, name="model")
    return model


def get_pretrained_weights(
    inputs: KerasTensor,
    checkpoint_file: str,
    task: EcgTask,
    arch: ArchitectureType = "resnet18",
    stages: Optional[int] = None,
) -> tf.keras.Model:
    """Initialize model with weights from file

    Args:
        checkpoint_file (str): TensorFlow checkpoint file containing weights
        task (EcgTask): Hear arrhythmia task
        arch (ArchitectureType, optional): Architecture type. Defaults to 'resnet18'.
        stages (Optional[int], optional): # stages in network. Defaults to None.

    Returns:
        tf.keras.Model: Pre-trained model
    """
    model = generate_task_model(
        task, arch, stages=stages, return_feature_extractor=True
    )
    if task in [EcgTask.rhythm, EcgTask.beat, EcgTask.hr]:
        inputs = build_input_tensor_from_shape(tf.TensorShape((None, 1)))
    else:
        raise ValueError("Unknown task: {}".format(task))
    model(inputs)
    model.load_weights(checkpoint_file)
    return model


def get_predicted_threshold_indices(
    y_prob: npt.ArrayLike,
    y_pred: Optional[npt.ArrayLike] = None,
    threshold: float = 0.5,
) -> npt.ArrayLike:
    """Get prediction indices that are above threshold (confidence level).
    This is useful to remove weak prediction that can happen due to noisy data or poor model performance.

    Args:
        y_prob (npt.ArrayLike): Model output as probabilities
        y_pred (npt.ArrayLike, optional): Model predictions. Defaults to None.
        threshold (float): Confidence level

    Returns:
        npt.ArrayLike: Indices of y_prob that satisfy threshold
    """
    if y_pred is None:
        y_pred = np.argmax(y_prob, axis=1)

    y_pred_prob = np.take_along_axis(
        y_prob, np.expand_dims(y_pred, axis=-1), axis=-1
    ).squeeze(axis=-1)
    y_thresh_idx = np.where(y_pred_prob > threshold)[0]
    return y_thresh_idx


def predict_tflite(
    model_content: bytes,
    test_x: npt.ArrayLike,
    input_name: Optional[str] = None,
    output_name: Optional[str] = None,
) -> npt.ArrayLike:
    """Perform prediction using tflite model content

    Args:
        model_content (bytes): TFLite model content
        test_x (npt.ArrayLike): Input dataset w/ no batch dimension
        input_name (Optional[str], optional): Input layer name. Defaults to None.
        output_name (Optional[str], optional): Output layer name. Defaults to None.

    Returns:
        npt.ArrayLike: Model outputs
    """
    # Prepare the test data
    inputs = test_x.copy()
    inputs = inputs.astype(np.float32)

    interpreter = tf.lite.Interpreter(model_content=model_content)
    model_sig = interpreter.get_signature_runner()
    inputs_details = model_sig.get_input_details()
    outputs_details = model_sig.get_output_details()
    if input_name is None:
        input_name = list(inputs_details.keys())[0]
    if output_name is None:
        output_name = list(outputs_details.keys())[0]
    input_details = inputs_details[input_name]
    output_details = outputs_details[output_name]
    input_scale: List[float] = input_details["quantization_parameters"]["scales"]
    input_zero_point: List[int] = input_details["quantization_parameters"][
        "zero_points"
    ]
    output_scale: List[float] = output_details["quantization_parameters"]["scales"]
    output_zero_point: List[int] = output_details["quantization_parameters"][
        "zero_points"
    ]

    if input_scale and input_zero_point:
        inputs = inputs / input_scale[0] + input_zero_point[0]
        inputs = inputs.astype(input_details["dtype"])

    outputs = np.array(
        [
            model_sig(**{input_name: inputs[i : i + 1]})[output_name][0]
            for i in range(inputs.shape[0])
        ],
        dtype=output_details["dtype"],
    )

    if output_scale and output_zero_point:
        outputs = outputs.astype(np.float32)
        outputs = (outputs - output_zero_point[0]) * output_scale[0]

    return outputs


def evaluate_tflite(
    model: tf.keras.Model,
    model_content: bytes,
    test_x: npt.ArrayLike,
    y_true: npt.ArrayLike,
) -> npt.ArrayLike:
    """Get loss values of TFLite model for given dataset

    Args:
        model (tf.keras.Model): TF model
        model_content (bytes): TFLite model
        test_x (npt.ArrayLike): Input samples
        y_true (npt.ArrayLike): Input labels

    Returns:
        npt.ArrayLike: Loss values
    """
    y_pred = predict_tflite(model_content, test_x=test_x)
    loss_function = tf.keras.losses.get(model.loss)
    loss = loss_function(y_true, y_pred).numpy()
    return loss
