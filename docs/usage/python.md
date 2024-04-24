# Python Usage

__HeartKit__ python package allows for more fine-grained control and customization. You can use the package to train, evaluate, and deploy models for a variety of tasks. The package is designed to be simple and easy to use.

!!! Example

    --8<-- "assets/usage/python-full-snippet.md"

---

## [Download](../modes/train.md)

The `download` command is used to download all datasets specified. Please refer to [Datasets](../datasets/index.md) for details on the available datasets. Additional datasets can be added by creating a new dataset class and registering it with __HeartKit dataset factory__.

!!! example "Python"

    The following snippet will download and prepare four datasets.

     --8<-- "assets/usage/python-download-snippet.md"

---

## [Train](../modes/train.md)

The `train` command is used to train a HeartKit model for the specified `task` and `dataset`. Each task provides a reference routine for training the model. The routine can be customized via the `hk.HKTrainParams` configuration. If further customization is needed, the task's routine can be overriden.

!!! example "Python"

    The following snippet will train a rhythm model using the supplied parameters:

    --8<-- "assets/usage/python-train-snippet.md"


---

## [Evaluate](../modes/evaluate.md)

The `evaluate` command will test the performance of the model on the reserved test set for the specified `task`. The routine can be customized via the `hk.HKTestParams` configuration. A number of results and metrics will be generated and saved to the `job_dir`.

!!! Example "Python"

    The following command will test the rhythm model using the supplied parameters:

    --8<-- "assets/usage/python-evaluate-snippet.md"

---

## [Export](../modes/export.md)

The `export` command will convert the trained TensorFlow model into both TensorFlow Lite (TFL) and TensorFlow Lite for micro-controller (TFLM) variants. The command will also verify the models' outputs match. The activations and weights can be quantized by configuring the `quantization` section in the configuration file. Once converted, the TFLM header file will be copied to location specified by `tflm_file`.

!!! Example "Python"

    The following command will export the rhythm model to TF Lite and TFLM:

    --8<-- "assets/usage/python-export-snippet.md"

---

## [Demo](../modes/demo.md)

The `demo` command is used to run a task-level demonstration using the designated backend inference engine (e.g. PC or EVB). The routine can be customized via the `hk.HKDemoParams` configuration. If running on the EVB, additional setup is required to flash and connect the EVB to the PC.

!!! Example "Python"

    The following snippet will run a task-level demo using the PC as the backend inference engine:

    --8<-- "assets/usage/python-demo-snippet.md"

---
