# Model Exporting

## <span class="sk-h2-span">Introduction </span>

Export mode is used to convert the trained TensorFlow model into a format that can be used for deployment onto Ambiq's family of SoCs. Currently, the command will convert the TensorFlow model into both TensorFlow Lite (TFL) and TensorFlow Lite for micro-controller (TFLM) variants. The command will also verify the models' outputs match. The activations and weights can be quantized by configuring the `quantization` section in the configuration file or by setting the `quantization` parameter in the code.

---
## <span class="sk-h2-span">Usage</span>

!!! Example

    The following command will export the rhythm model to TF Lite and TFLM:

    === "CLI"

        ```bash
        heartkit --mode export --task rhythm --config ./configs/rhythm-class-2.json
        ```

    === "Python"

        --8<-- "assets/modes/python-export-snippet.md"

---

## <span class="sk-h2-span">Arguments </span>

Please refer to [HKExportParams](../modes/configuration.md#hkexportparams) for the list of arguments that can be used with the `evaluate` command.
