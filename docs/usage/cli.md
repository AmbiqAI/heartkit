# :octicons-terminal-24: HeartKit CLI

<div class="termy">

```console
$ heartkit --help

HeartKit CLI Options:
    --task [segmentation, rhythm, beat, denoise]
    --mode [download, train, evaluate, export, demo]
    --config ["./path/to/config.json", or '{"raw: "json"}']
```

</div>

The HeartKit command line interface (CLI) makes it easy to run a variety of single-line commands without the need for writing any code. You can run all built-in tasks and modes from the terminal with the `heartkit` command. This is also useful to reproduce [Model Zoo](../zoo/index.md) results.

!!! example

    === "Syntax"
        Heartkit commands use the following syntax:

        ```bash
        heartkit --mode [MODE] --task [TASK] --config [CONFIG]
        ```

        Or using short flags:

        ```bash
        heartkit -m [MODE] -t [TASK] -c [CONFIG]
        ```

        Where:

        * `MODE` is one of `download`, `train`, `evaluate`, `export`, or `demo`
        * `TASK` is one of `segmentation`, `rhythm`, `beat`, or `denoise`
        * `CONFIG` is configuration as JSON content or file path

    === "Download"
        Download datasets specified in the configuration file.

        ```bash
        heartkit -m download -c ./configs/download-datasets.json
        ```

    === "Train"
        Train a rhythm model using the supplied configuration file.

        ```bash
        heartkit -m train -t rhythm -c ./configuration.json
        ```

    === "Evaluate"
        Evaluate the rhythm model using the supplied configuration file.

        ```bash
        heartkit -m evaluate -t rhythm  -c ./configuration.json
        ```

    === "Demo"
        Run demo on trained rhythm model using the supplied configuration file.

        ```bash
        heartkit -m demo -t rhythm -c ./configuration.json
        ```


!!! Note "Configuration File"

    The configuration file is a JSON file that contains all the necessary parameters for the task. The configuration file can be passed as a file path or as a JSON string. In addition, a single configuration file can be used for all `modes`- only needed parameters will be extracted for the given `mode` running.  Please refer to the [Configuration](../modes/configuration.md) section for more details.

---

## [Download](../modes/download.md)

The [download](../modes/download.md) command is used to download all datasets specified in the configuration file. Please refer to [Datasets](../datasets/index.md) for details on the available datasets.


!!! Example "CLI"

    The following command will download and prepare all datasets specified in configuration JSON file.

    ```bash
    heartkit --task rhythm --mode download --config ./configs/download-datasets.json
    ```

---

## [Train](../modes/train.md)

The [train](../modes/train.md) command is used to train a HeartKit model for the specified `task` and `dataset`. Each task provides a reference routine for training the model. The routine can be customized via the configuration file. Please refer to [HKTaskParams](../modes/configuration.md#hktaskparams) to see supported options.

!!! Example "CLI"

    The following command will train a rhythm model using the reference configuration:

    ```bash
    heartkit --task rhythm --mode train --config ./configuration.json
    ```

---

## [Evaluate](../modes/evaluate.md)

The [evaluate](../modes/evaluate.md) command will test the performance of the model on the reserved test sets for the specified `task`. The routine can be customized via the configuration file. Please refer to [HKTaskParams](../modes/configuration.md#hktaskparams) to see supported options.

!!! example "CLI"

    The following command will test the rhythm model using the reference configuration:

    ```bash
    heartkit --task rhythm --mode evaluate --config ./configuration.json
    ```

---

## [Export](../modes/export.md)

The [export](../modes/export.md) command will convert the trained TensorFlow model into both TensorFlow Lite (TFL) and TensorFlow Lite for micro-controller (TFLM) variants. The command will also verify the models' outputs match. The activations and weights can be quantized by configuring the `quantization` section in the configuration file. Once converted, the TFLM header file will be copied to location specified by `tflm_file`.

!!! example "CLI"

    The following command will export the rhythm model to TF Lite and TFLM:

    ```bash
    heartkit --task rhythm --mode export --config ./configuration.json
    ```

---

## [Demo](../modes/demo.md)


The [demo](../modes/demo.md) command is used to run a task-level demonstration using either the PC or EVB as backend inference engine.

!!! Example "CLI"

    The following command will run a demo on the trained rhythm model using the same supplied configuration file.

    ```bash
    heartkit --task rhythm --mode demo --config ./configuration.json
    ```

---
