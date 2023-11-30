# Overview

__HeartKit__ can be used as either a CLI-based app or as a python package to perform advanced experimentation. In both forms, HeartKit exposes a number of modes and tasks discussed below:

---

## Modes

* `download`: Download datasets
* `train`: Train a model for specified task and dataset(s)
* `evaluate`: Evaluate a model for specified task and dataset(s)
* `export`: Export a trained model to TensorFlow Lite and TFLM
* `demo`: Run task-level demo on PC or EVB

---

!!! Tasks

    === "segmentation"

        ### ECG Segmentation

        Delineate ECG signal into individual waves (P, QRS, T). <br>
        Refer to [Segmentation Task](./segmentation/overview.md) for more details.

    === "arrhythmia"

        ### Arrhythmia Classification

        Identify rhythm-level arrhythmias such as AFIB and AFL. <br>
        Refer to [Arrhythmia Task](./arrhythmia/overview.md) for more details.


    === "beat"

        ### Beat Classification

        Identify premature and escape beats. <br>
        Refer to [Beat Task](./beat/overview.md) for more details.


---

## Using CLI

The HeartKit command line interface (CLI) makes it easy to run a variefy of single-line commands without the need for writing any code. You can rull all tasks and modes from the terminal with the `heartkit` command.

<div class="termy">

```console
$ heartkit --help

HeartKit CLI Options:
    --task [segmentation, arrhythmia, beat]
    --mode [download, train, evaluate, export, demo]
    --config ["./path/to/config.json", or '{"raw: "json"}']
```

</div>


!!! note
    Before running commands, be sure to activate python environment: `poetry shell`. On Windows using Powershell, use `.venv\Scripts\activate.ps1`.

## __1. Download Datasets__

The `download` command is used to download all datasets specified in the configuration file. Please refer to [Datasets](./datasets.md) for details on the available datasets.

The following example will download and prepare all currently used datasets.

!!! example

    === "CLI"

        ```bash
        heartkit --mode download --config ./configs/download-datasets.json
        ```

    === "Python"

        ```python
        import heartkit as hk

        hk.datasets.download_datasets(hk.defines.HeartDownloadParams(
            ds_path="./datasets",
            datasets=["icentia11k", "ludb", "qtdb", "synthetic"],
            progress=True
        ))
        ```

???+ note
    The __Icentia11k dataset__ requires roughly 200 GB of disk space and can take around 2 hours to download.

## __2. Train Model__

The `train` command is used to train a HeartKit model for the specified `task` and `dataset`. Please refer to `heartkit/defines.py` to see supported options.

!!! example

    The following command will train a 2-class arrhythmia model using the reference configuration:

    === "CLI"

        ```bash
        heartkit --task arrhythmia --mode train --config ./configs/train-arrhythmia-model.json
        ```

    === "Python"

        ```python
        import heartkit as hk

        hk.arrhythmia.train(hk.defines.HeartTrainParams(
            job_dir="./results/arrhythmia",
            ds_path="./datasets",
            sampling_rate=200,
            frame_size=800,
            num_classes=2,
            samples_per_patient=[100, 800],
            val_samples_per_patient=[100, 800],
            train_patients=10000,
            val_patients=0.10,
            val_size=200000,
            batch_size=256,
            buffer_size=100000,
            epochs=100,
            steps_per_epoch=20,
            val_metric="loss",
            datasets=["icentia11k"]
        ))
        ```

## __3. Evaluate Model__

The `evaluate` command will evaluate the performance of the model on the reserved test set for the specified `task`. For certain tasks, a confidence threshold can also be set such that a label is only assigned when the model's probability is greater than the threshold; otherwise, a label of inconclusive will be assigned.

!!! example

    The following command will test the 2-class arrhythmia model using the reference configuration:

    === "CLI"

        ```bash
        heartkit --task arrhythmia --mode evaluate --config ./configs/evaluate-arrhythmia-model.json
        ```

    === "Python"

        ```python
        import heartkit as hk

        hk.arrhythmia.evaluate(hk.defines.HeartTestParams(
            job_dir="./results/arrhythmia",
            ds_path="./datasets",
            sampling_rate=200,
            frame_size=800,
            num_classes=2,
            samples_per_patient=[100, 800],
            test_patients=1000,
            test_size=100000,
            model_file="./results/arrhythmia/model.tf",
            threshold=0.75
        ))
        ```

## __4. Export Model__

The `export` command will convert the trained TensorFlow model into both TensorFlow Lite (TFL) and TensorFlow Lite for microcontroller (TFLM) variants. The command will also verify the models' outputs match. Post-training quantization can also be enabled by setting the `quantization` flag in the configuration.

!!! example

    The following command will export the 2-class arrhythmia model to TF Lite and TFLM:

    === "CLI"

        ```bash
        heartkit --task arrhythmia --mode export --config ./configs/export-arrhythmia-model.json
        ```

    === "Python"

        ```python
        import heartkit as hk

        hk.arrhythmia.export(hk.defines.HeartExportParams(
            job_dir="./results/arrhythmia",
            ds_path="./datasets",
            sampling_rate=200,
            frame_size=800,
            num_classes=2,
            samples_per_patient=[100, 500, 100],
            model_file="./results/arrhythmia/model.tf",
            quantization=true,
            threshold=0.95,
            tflm_var_name="g_arrhythmia_model",
            tflm_file="./evb/src/arrhythmia_model_buffer.h"
        ))
        ```

Once converted, the TFLM header file will be copied to location specified by `tflm_file`. If parameters were changed (e.g. window size, quantization), `./evb/src/constants.h` will need to be updated accordingly.

## __5. Task Demo__

The `demo` command is used to run a task-level demonstration using either the PC or EVB as backend inference engine.

!!! example

    === "CLI"

        ```bash
        heartkit --task arrhythmia --mode demo --config ./configs/demo-arrhythmia.json
        ```

    === "Python"

        ```python
        import heartkit as hk

        hk.arrhythmia.demo(hk.defines.HKDemoParams(
            job_dir="./results/arrhythmia",
            ds_path="./datasets",
            sampling_rate=200,
            frame_size=800,
            num_classes=2,
            model_file="./results/arrhythmia/model.tflite",
            backend="pc"
        ))
        ```
