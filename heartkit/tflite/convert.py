import io
import os

import keras
import numpy as np
import numpy.typing as npt
import pandas as pd
import tensorflow as tf


def array_dump(
    data: npt.NDArray,
    dst_path: os.PathLike,
    var_name: str = "test_stimulus",
    var_dtype: str | None = None,
    row_len: int = 12,
    is_header: bool = False,
):
    """Generate C array of values from flattened numpy array.

    Args:
        data (npt.NDArray): Data array
        dst_path (PathLike): C file destination path
        var_name (str, optional): C variable name. Defaults to "test_stimulus".
        var_dtype (str | None, optional): C variable type. Defaults to None.
        row_len (int, optional): Elements to write per row. Defaults to 12.
        is_header (bool): Write as header or source C file. Defaults to source.
    """
    data = data.flatten()

    if isinstance(data[0], np.floating):
        var_dtype = "float"
    elif isinstance(data[0], np.int8):
        var_dtype = "int8_t"
    elif isinstance(data[0], np.int16):
        var_dtype = "int16_t"
    elif isinstance(data[0], np.integer):
        var_dtype = "int32_t"
    else:
        raise ValueError("Unsupported dtype")

    with open(dst_path, "w", encoding="UTF-8") as wfp:
        if is_header:
            wfp.write(f"#ifndef __{var_name.upper()}_H{os.linesep}")
            wfp.write(f"#define __{var_name.upper()}_H{os.linesep}")

        wfp.write(f"#include <cstdint>{os.linesep}{os.linesep}")

        wfp.write(f"const {var_dtype} {var_name}[] = {{{os.linesep}")
        for row in range(0, len(data), row_len):
            wfp.write("  " + ", ".join((str(val) for val in data[row : row + row_len])) + f", {os.linesep}")
        # END FOR
        wfp.write(f"}};{os.linesep}")
        wfp.write(f"const unsigned int {var_name}_len = {len(data)};{os.linesep}")
        if is_header:
            wfp.write(f"#endif // __{var_name.upper()}_H{os.linesep}")
    # END WITH


def xxd_c_dump(
    src_path: os.PathLike,
    dst_path: os.PathLike,
    var_name: str = "tflm_model",
    chunk_len: int = 12,
    is_header: bool = False,
):
    """Generate C like char array of hex values from binary source. Equivalent to `xxd -i src_path > dst_path`
        but with added features to provide # columns and variable name.

    Args:
        src_path (PathLike): Binary file source path
        dst_path (PathLike): C file destination path
        var_name (str, optional): C variable name. Defaults to 'g_model'.
        chunk_len (int, optional): # of elements per row. Defaults to 12.
        is_header (bool): Write as header or source C file. Defaults to source.
    """
    var_len = 0
    with open(src_path, "rb", encoding=None) as rfp, open(dst_path, "w", encoding="UTF-8") as wfp:
        if is_header:
            wfp.write(f"#ifndef __{var_name.upper()}_H{os.linesep}")
            wfp.write(f"#define __{var_name.upper()}_H{os.linesep}")

        wfp.write(f"const unsigned char {var_name}[] = {{{os.linesep}")
        for chunk in iter(lambda: rfp.read(chunk_len), b""):
            wfp.write("  " + ", ".join((f"0x{c:02x}" for c in chunk)) + f", {os.linesep}")
            var_len += len(chunk)
        # END FOR
        wfp.write(f"}};{os.linesep}")
        wfp.write(f"const unsigned int {var_name}_len = {var_len};{os.linesep}")
        if is_header:
            wfp.write(f"#endif // __{var_name.upper()}_H{os.linesep}")
    # END WITH


def convert_tflite(
    model: keras.Model,
    quantize: bool = False,
    test_x: npt.NDArray | None = None,
    input_type: str | None = None,
    output_type: str | None = None,
    supported_ops: list | None = None,
) -> bytes:
    """Convert TF model into TFLite model content

    Args:
        model (keras.Model): TF model
        quantize (bool, optional): Enable PTQ. Defaults to False.
        test_x (npt.NDArray | None, optional): Enables full integer PTQ. Defaults to None.
        input_type (str | None): Input type data format. Defaults to None.
        output_type (str | None): Output type data format. Defaults to None.

    Returns:
        bytes: TFLite content

    """

    converter = tf.lite.TFLiteConverter.from_keras_model(model=model)

    # Optionally quantize model
    if quantize:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

        if test_x is not None:
            converter.target_spec.supported_ops = supported_ops or [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = tf.dtypes.as_dtype(input_type) if input_type else None
            converter.inference_output_type = tf.dtypes.as_dtype(output_type) if output_type else None

            def rep_dataset():
                for i in range(test_x.shape[0]):
                    yield [test_x[i : i + 1]]

            converter.representative_dataset = rep_dataset
        # END IF
    # END IF

    # Convert model
    return converter.convert()


def debug_quant_tflite(
    model: keras.Model,
    test_x: npt.NDArray | None = None,
    input_type: str | None = None,
    output_type: str | None = None,
    supported_ops: list | None = None,
) -> tuple[tf.lite.experimental.QuantizationDebugger, pd.DataFrame]:
    """Debug quantized TFLite model content

    Args:
        model (keras.Model): TF model
        quantize (bool, optional): Enable PTQ. Defaults to False.
        test_x (npt.NDArray | None, optional): Enables full integer PTQ. Defaults to None.
        input_type (str | None): Input type data format. Defaults to None.
        output_type (str | None): Output type data format. Defaults to None.

    Returns:
        tuple[tf.lite.experimental.QuantizationDebugger, pd.DataFrame]: TFlite debugger, Layer statistics

    """
    converter = tf.lite.TFLiteConverter.from_keras_model(model=model)

    def rep_dataset():
        for i in range(test_x.shape[0]):
            yield [test_x[i : i + 1]]

    # Quantize model
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = supported_ops or [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.dtypes.as_dtype(input_type) if input_type else None
    converter.inference_output_type = tf.dtypes.as_dtype(output_type) if output_type else None
    converter.representative_dataset = rep_dataset

    # Debug model
    debugger = tf.lite.experimental.QuantizationDebugger(converter=converter, debug_dataset=rep_dataset)
    debugger.run()

    with io.StringIO() as f:
        debugger.layer_statistics_dump(f)
        f.seek(0)
        layer_stats = pd.read_csv(f)

    # Add custom metrics
    layer_stats["range"] = 255.0 * layer_stats["scale"]
    layer_stats["rmse/scale"] = layer_stats.apply(lambda row: np.sqrt(row["mean_squared_error"]) / row["scale"], axis=1)
    return debugger, layer_stats


def predict_tflite(
    model_content: bytes,
    test_x: npt.NDArray,
    input_name: str | None = None,
    output_name: str | None = None,
) -> npt.NDArray:
    """Perform prediction using tflite model content

    Args:
        model_content (bytes): TFLite model content
        test_x (npt.NDArray): Input dataset w/ no batch dimension
        input_name (str | None, optional): Input layer name. Defaults to None.
        output_name (str | None, optional): Output layer name. Defaults to None.

    Returns:
        npt.NDArray: Model outputs
    """
    # Prepare the test data
    inputs = test_x.copy()
    inputs = inputs.astype(np.float32)

    interpreter = tf.lite.Interpreter(model_content=model_content)
    interpreter.allocate_tensors()
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
    input_zero_point: list[int] = input_details["quantization_parameters"]["zero_points"]
    output_scale: list[float] = output_details["quantization_parameters"]["scales"]
    output_zero_point: list[int] = output_details["quantization_parameters"]["zero_points"]

    inputs = inputs.reshape([-1] + input_details["shape_signature"].tolist()[1:])
    if len(input_scale) and len(input_zero_point):
        inputs = inputs / input_scale[0] + input_zero_point[0]
        inputs = inputs.astype(input_details["dtype"])

    outputs = np.array(
        [model_sig(**{input_name: inputs[i : i + 1]})[output_name][0] for i in range(inputs.shape[0])],
        dtype=output_details["dtype"],
    )

    if len(output_scale) and len(output_zero_point):
        outputs = outputs.astype(np.float32)
        outputs = (outputs - output_zero_point[0]) * output_scale[0]

    return outputs


def evaluate_tflite(
    model: bytes,
    model_content: bytes,
    test_x: npt.NDArray,
    y_true: npt.NDArray,
) -> npt.NDArray:
    """Get loss values of TFLite model for given dataset

    Args:
        model (bytes): TFLite model bytes
        model_content (bytes): TFLite model
        test_x (npt.NDArray): Input samples
        y_true (npt.NDArray): Input labels

    Returns:
        npt.NDArray: Loss values
    """
    y_pred = predict_tflite(model_content, test_x=test_x)
    loss_function = keras.losses.get(model.loss)
    loss = loss_function(y_true, y_pred).numpy()
    return loss
