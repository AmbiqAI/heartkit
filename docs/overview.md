# Overview

__HeartKit__ can be used as either a CLI-based app or as a python package to perform advanced experimentation. In both forms, HeartKit exposes a number of modes and tasks discussed below:


## Modes

* `download`: Download datasets
* `train`: Train a model for specified task and dataset(s)
* `evaluate`: Evaluate a model for specified task and dataset(s)
* `export`: Export a trained model to TensorFlow Lite and TFLM
* `demo`: Run full demo on PC or EVB

## Tasks

* `Segmentation`: Perform ECG based segmentation (P-Wave, QRS, T-Wave)
* `HRV`: Heart rate, rhythm, HRV metrics (RR interval)
* `Arrhythmia`: Heart arrhythmia detection (AFIB, AFL)
* `Beat`: Classify individual beats (PAC, PVC)

## Using CLI

The HeartKit command line interface (CLI) makes it easy to run a variefy of single-line commands without the need for writing any code. You can rull all tasks and modes from the terminal with the `heartkit` command.

<div class="termy">

```console
$ heartkit --help

HeartKit CLI Options:
    --task [segmentation, arrhythmia, beat, hrv]
    --mode [download, train, evaluate, export, demo]
    --config ["./path/to/config.json", or '{"raw: "json"}']
```

</div>

<!-- ```bash
heartkit
--task [segmentation, arrhythmia, beat, hrv]
--mode [download, train, evaluate, export, demo]
--config ["./path/to/config.json", or '{"raw: "json"}']
``` -->

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

!!! note
    The __Icentia11k dataset__ requires roughly 200 GB of disk space and can take around 2 hours to download.

## __2. Train Model__

The `train` command is used to train a HeartKit model. The following command will train the arrhythmia model using the reference configuration. Please refer to `heartkit/defines.py` to see supported options.

!!! example

    === "CLI"

        ```bash
        heartkit --task arrhythmia --mode train --config ./configs/train-arrhythmia-model.json
        ```

    === "Python"

        ```python
        import heartkit as hk

        hk.arrhythmia.train_model(hk.defines.HeartTrainParams(
            job_dir="./results/arrhythmia",
            ds_path="./datasets",
            sampling_rate=200,
            frame_size=800,
            samples_per_patient=[100, 800, 800],
            val_samples_per_patient=[100, 800, 800],
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

The `evaluate` command will evaluate the performance of the model on the reserved test set. A confidence threshold can also be set such that a label is only assigned when the model's probability is greater than the threshold; otherwise, a label of inconclusive will be assigned.

!!! example

    === "CLI"

        ```bash
        heartkit --task arrhythmia --mode evaluate --config ./configs/evaluate-arrhythmia-model.json
        ```

    === "Python"

        ```python
        import heartkit as hk

        hk.arrhythmia.evaluate_model(hk.defines.HeartTestParams(
            job_dir="./results/arrhythmia",
            ds_path="./datasets",
            sampling_rate=200,
            frame_size=800,
            samples_per_patient=[100, 800, 800],
            test_patients=1000,
            test_size=100000,
            model_file="./results/arrhythmia/model.tf",
            threshold=0.95
        ))
        ```

## __4. Export Model__

The `export` command will convert the trained TensorFlow model into both TensorFlow Lite (TFL) and TensorFlow Lite for microcontroller (TFLM) variants. The command will also verify the models' outputs match. Post-training quantization can also be enabled by setting the `quantization` flag in the configuration.

!!! example

    === "CLI"

        ```bash
        heartkit --task arrhythmia --mode export --config ./configs/export-arrhythmia-model.json
        ```

    === "Python"

        ```python
        import heartkit as hk

        hk.arrhythmia.export_model(hk.defines.HeartExportParams(
            job_dir="./results/arrhythmia",
            ds_path="./datasets",
            sampling_rate=200,
            frame_size=800,
            samples_per_patient=[100, 500, 100],
            model_file="./results/arrhythmia/model.tf",
            quantization=true,
            threshold=0.95,
            tflm_var_name="g_arrhythmia_model",
            tflm_file="./evb/src/arrhythmia_model_buffer.h"
        ))
        ```

Once converted, the TFLM header file will be copied to location specified by `tflm_file`. If parameters were changed (e.g. window size, quantization), `./evb/src/constants.h` will need to be updated accordingly.

## __5. Demo__

The `demo` command is used to run a full-fledged HeartKit demonstration. The demo is decoupled into three tasks: (1) a REST server to provide a unified API, (2) a front-end UI, and (3) a backend to fetch samples and perform inference. The host PC performs tasks (1) and (2). For (3), the trained models can run on either the `PC` or an Apollo 4 evaluation board (`EVB`) by setting the `backend` field in the configuration. When the `PC` backend is selected, the host PC will perform task (3) entirely to fetch samples and perform inference. When the `EVB` backend is selected, the `EVB` will perform inference using either sensor data or prior data. The PC connects to the `EVB` via RPC over serial transport to provide sample data and capture inference results.

Please refer to [Arrhythmia Demo Tutorial](./tutorials/arrhythmia-demo.md) and [HeartKit Demo Tutorial](./tutorials/heartkit-demo.md) for further instructions.
