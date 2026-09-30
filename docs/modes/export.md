# :material-export: Model Exporting

## Introduction

Export mode is used to convert the trained TensorFlow model into a format that can be used for deployment onto Ambiq's family of SoCs. Currently, the command will convert the TensorFlow model into both TensorFlow Lite (TFL) and TensorFlow Lite for micro-controller (TFLM) variants. The command will also verify the models' outputs match. The activations and weights can be quantized by configuring the `quantization` section in the configuration file or by setting the `quantization` parameter in the code.

<div class="annotate" markdown>

1. Load the configuration data (e.g. `configuration.json`)
1. Load the test data (e.g. `test.pkl`)
1. Load the trained model (e.g. `model.keras`)
1. Quantize the model (e.g. `16x8`)
1. Convert the model (e.g. `TFL`, `TFLM`)
1. Verify the models' outputs match
1. Save artifacts (e.g. `model.tflite`)

</div>

**Example configuration**
--8<-- "assets/usage/json-configuration.md"


```mermaid
flowchart TD
    A["Load configuration and test data"] --> B
    B["Load trained model"] --> C
    C["Quantize model"] --> D
    D["Convert model"] --> E
    E["Verify outputs"] --> F
    F["Save deployment artifacts"]
```

---

## Usage

### CLI

The following command will export a rhythm model using the reference configuration.

```bash
heartkit --task rhythm --mode export --config ./configuration.json
```

### Python

The model can be evaluated using the following snippet:

```py linenums="1"

task = hk.TaskFactory.get("rhythm")

params = hk.HKTaskParams(...)

task.export(params)

```

**Example configuration**
--8<-- "assets/usage/python-configuration.md"

---

## Arguments

Please refer to [HKTaskParams](../modes/configuration.md#hktaskparams) for the list of arguments that can be used with the `export` command.
