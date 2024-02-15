# CLI Usage

<div class="termy">

```console
$ heartkit --help

HeartKit CLI Options:
    --task [segmentation, arrhythmia, beat, denoise]
    --mode [download, train, evaluate, export, demo]
    --config ["./path/to/config.json", or '{"raw: "json"}']
```

</div>

The HeartKit command line interface (CLI) makes it easy to run a variety of single-line commands without the need for writing any code. You can run all tasks and modes from the terminal with the `heartkit` command.

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
        * `TASK` is one of `segmentation`, `arrhythmia`, `beat`, or `denoise`
        * `CONFIG` is configuration as JSON content or file path

    === "Download"
        Download datasets specified in the configuration file.

        ```bash
        heartkit -m download -c ./configs/download-datasets.json
        ```

    === "Train"
        Train a 2-class arrhythmia model using the supplied configuration file.

        ```bash
        heartkit -m train -t arrhythmia -c ./configs/arrhythmia-class-2.json
        ```

    === "Evaluate"
        Evaluate the trained arrhythmia model using the supplied configuration file. Note that we are using the same configuration file as the training command- we can store all the parameters in the same file. Only needed parameters will be used for each command.

        ```bash
        heartkit -m evaluate -t arrhythmia  -c ./configs/arrhythmia-class-2.json
        ```

    === "Demo"
        Run demo on trained arrhythmia model using the supplied configuration file.

        ```bash
        heartkit -m demo -t arrhythmia -c ./configs/arrhythmia-class-2.json
        ```

---

## [Download](../modes/download.md)

The `download` command is used to download all datasets specified in the configuration file. Please refer to [Datasets](../datasets/index.md) for details on the available datasets.


!!! Example "CLI"

    The following command will download and prepare all datasets specified in configuration JSON file.

    ```bash
    heartkit --mode download --config ./configs/download-datasets.json
    ```

---

## [Train](../modes/train.md)

The `train` command is used to train a HeartKit model for the specified `task` and `dataset`. Please refer to `heartkit/defines.py` to see supported options.

!!! Example "CLI"

    The following command will train a 2-class arrhythmia model using the reference configuration:

    ```bash
    heartkit --task arrhythmia --mode train --config ./configs/arrhythmia-class-2.json
    ```

---

## [Evaluate](../modes/evaluate.md)

The `evaluate` command will test the performance of the model on the reserved test set for the specified `task`. For certain tasks, a confidence threshold can also be set such that a label is only assigned when the model's probability is greater than the threshold; otherwise, a label of inconclusive will be assigned. This is useful in noisy environments where the model may not be confident in its prediction.

!!! example "CLI"

    The following command will test the 2-class arrhythmia model using the reference configuration:

    ```bash
    heartkit --task arrhythmia --mode evaluate --config ./configs/arrhythmia-class-2.json
    ```

---

## [Export](../modes/export.md)

The `export` command will convert the trained TensorFlow model into both TensorFlow Lite (TFL) and TensorFlow Lite for micro-controller (TFLM) variants. The command will also verify the models' outputs match. Post-training quantization (PTQ) can also be enabled by setting the `quantization` flag in the configuration. Once converted, the TFLM header file will be copied to location specified by `tflm_file`.

!!! example "CLI"

    The following command will export the 2-class arrhythmia model to TF Lite and TFLM:


    ```bash
    heartkit --task arrhythmia --mode export --config ./configs/arrhythmia-class-2.json
    ```

---

## [Demo](../modes/demo.md)


The `demo` command is used to run a task-level demonstration using either the PC or EVB as backend inference engine.

!!! Example "CLI"

    The following command will run a demo on the trained arrhythmia model using the same supplied configuration file.

    ```bash
    heartkit --task arrhythmia --mode demo --config ./configs/arrhythmia-class-2.json
    ```

---
