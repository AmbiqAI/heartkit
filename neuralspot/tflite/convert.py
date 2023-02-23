import os

import numpy as np
import numpy.typing as npt
import tensorflow as tf


def xxd_c_dump(
    src_path: str,
    dst_path: str,
    var_name: str = "tflm_model",
    chunk_len: int = 12,
    is_header: bool = False,
):
    """Generate C like char array of hex values from binary source. Equivalent to `xxd -i src_path > dst_path`
        but with added features to provide # columns and variable name.
    Args:
        src_path (str): Binary file source path
        dst_path (str): C file destination path
        var_name (str, optional): C variable name. Defaults to 'g_model'.
        chunk_len (int, optional): # of elements per row. Defaults to 12.
    """
    var_len = 0
    with open(src_path, "rb", encoding=None) as rfp, open(
        dst_path, "w", encoding="UTF-8"
    ) as wfp:
        if is_header:
            wfp.write(f"#ifndef __{var_name.upper()}_H{os.linesep}")
            wfp.write(f"#define __{var_name.upper()}_H{os.linesep}")

        wfp.write(f"const unsigned char {var_name}[] = {{{os.linesep}")
        for chunk in iter(lambda: rfp.read(chunk_len), b""):
            wfp.write(
                "  " + ", ".join((f"0x{c:02x}" for c in chunk)) + f", {os.linesep}"
            )
            var_len += len(chunk)
        # END FOR
        wfp.write(f"}};{os.linesep}")
        wfp.write(f"const unsigned int {var_name}_len = {var_len};{os.linesep}")
        if is_header:
            wfp.write(f"#endif // __{var_name.upper()}_H{os.linesep}")
    # END WITH


def convert_tflite(
    model: tf.keras.Model,
    quantize: bool = False,
    test_x: npt.ArrayLike | None = None,
    input_type: tf.DType | None = None,
    output_type: tf.DType | None = None,
) -> bytes:
    """Convert TF model into TFLite model content

    Args:
        model (tf.keras.Model): TF model
        quantize (bool, optional): Enable PTQ. Defaults to False.
        test_x (npt.ArrayLike | None, optional): Enables full integer PTQ. Defaults to None.
        input_type (tf.DType | None): Input type data format. Defaults to None.
        output_type (tf.DType | None): Output type data format. Defaults to None.

    Returns:
        bytes: TFLite content

    """

    converter = tf.lite.TFLiteConverter.from_keras_model(model=model)

    # Optionally quantize model
    if quantize:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        if test_x is not None:
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = input_type
            converter.inference_output_type = output_type

            def rep_dataset():
                for i in range(test_x.shape[0]):
                    yield [test_x[i : i + 1]]

            converter.representative_dataset = rep_dataset
        # END IF
    # Convert model
    return converter.convert()


def predict_tflite(
    model_content: bytes,
    test_x: npt.ArrayLike,
    input_name: str | None = None,
    output_name: str | None = None,
) -> npt.ArrayLike:
    """Perform prediction using tflite model content

    Args:
        model_content (bytes): TFLite model content
        test_x (npt.ArrayLike): Input dataset w/ no batch dimension
        input_name (str | None, optional): Input layer name. Defaults to None.
        output_name (str | None, optional): Output layer name. Defaults to None.

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
    input_scale: list[float] = input_details["quantization_parameters"]["scales"]
    input_zero_point: list[int] = input_details["quantization_parameters"][
        "zero_points"
    ]
    output_scale: list[float] = output_details["quantization_parameters"]["scales"]
    output_zero_point: list[int] = output_details["quantization_parameters"][
        "zero_points"
    ]

    if len(input_scale) and len(input_zero_point):
        inputs = inputs / input_scale[0] + input_zero_point[0]
        inputs = inputs.astype(input_details["dtype"])

    outputs = np.array(
        [
            model_sig(**{input_name: inputs[i : i + 1]})[output_name][0]
            for i in range(inputs.shape[0])
        ],
        dtype=output_details["dtype"],
    )

    if len(output_scale) and len(output_zero_point):
        outputs = outputs.astype(np.float32)
        outputs = (outputs - output_zero_point[0]) * output_scale[0]

    print(input_zero_point, input_scale, output_zero_point, output_scale)
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
