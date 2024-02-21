# Python Usage

__HeartKit__ python package allows for more fine-grained control and customization. You can use the package to train, evaluate, and deploy models for a variety of tasks. The package is designed to be simple and easy to use.

!!! Example

    ```python
    import heartkit as hk

    ds_params = hk.HKDownloadParams.parse_file("download-datasets.json")
    train_params = hk.HKTrainParams.parse_file("train.json")
    test_params = hk.HKTestParams.parse_file("evaluate.json")
    export_params = hk.HKExportParams.parse_file("export.json")

    # Download datasets
    hk.datasets.download_datasets(ds_params)

    task = hk.TaskFactory.get("arrhythmia")

    # Train arrhythmia model
    task.train(train_params)

    # Evaluate arrhythmia model
    task.evaluate(test_params)

    # Export arrhythmia model
    task.export(export_params)

    ```

---

## [Download](../modes/train.md)

The `download` command is used to download all datasets specified. Please refer to [Datasets](../datasets/index.md) for details on the available datasets. Additional datasets can be added by creating a new dataset class and registering it with __HeartKit__ dataset factory.

!!! example "Python"

    The following snippet will download and prepare four datasets.

    ```python
    from pathlib import Path
    import heartkit as hk

    hk.datasets.download_datasets(hk.HKDownloadParams(
        ds_path=Path("./datasets"),
        datasets=["icentia11k", "ludb", "qtdb", "synthetic"],
        progress=True
    ))
    ```

---

## [Train](../modes/train.md)

The `train` command is used to train a HeartKit model for the specified `task` and `dataset`. Each task provides a reference routine for training the model. The routine can be customized by providing a configuration file or by setting the parameters directly in the code.

!!! example "Python"

    The following snippet will train a 2-class arrhythmia model using the reference configuration:

    ```python
    from pathlib import Path
    import heartkit as hk

    task = hk.TaskFactory.get("arrhythmia")

    task.train(hk.HKTrainParams(
        job_dir=Path("./results/arrhythmia-class-2"),
        ds_path=Path("./datasets"),
        datasets=[{
            "name": "icentia11k",
            "params": {}
        }],
        num_classes=2,
        class_map={
            0: 0,
            1: 1,
            2: 1
        },
        class_names=[
            "NONE", "AFIB/AFL"
        ],
        sampling_rate=200,
        frame_size=800,
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
    ))
    ```

---

## [Evaluate](../modes/evaluate.md)

The `evaluate` command will test the performance of the model on the reserved test set for the specified `task`. For certain tasks, a confidence threshold can also be set such that a label is only assigned when the model's probability is greater than the threshold; otherwise, a label of inconclusive will be assigned. This is useful in noisy environments where the model may not be confident in its prediction.

!!! Example "Python"

    The following command will test the 2-class arrhythmia model using the reference configuration:

    ```python
    from pathlib import Path
    import heartkit as hk

    task = hk.TaskFactory.get("arrhythmia")

    task.evaluate(hk.HKTestParams(
        job_dir=Path("./results/arrhythmia-class-2"),
        ds_path=Path("./datasets"),
        datasets=[{
            "name": "icentia11k",
            "params": {}
        }],
        num_classes=2,
        class_map={
            0: 0,
            1: 1,
            2: 1
        },
        class_names=[
            "NONE", "AFIB/AFL"
        ],
        sampling_rate=200,
        frame_size=800,
        test_samples_per_patient=[100, 800],
        test_patients=1000,
        test_size=100000,
        model_file=Path("./results/arrhythmia-class-2/model.keras"),
        threshold=0.75
    ))
    ```

---

## [Export](../modes/export.md)

The `export` command will convert the trained TensorFlow model into both TensorFlow Lite (TFL) and TensorFlow Lite for micro-controller (TFLM) variants. The command will also verify the models' outputs match. Post-training quantization (PTQ) can also be enabled by setting the `quantization` flag in the configuration. Once converted, the TFLM header file will be copied to location specified by `tflm_file`.

!!! Example "Python"

    The following command will export the 2-class arrhythmia model to TF Lite and TFLM:

    ```python
    from pathlib import Path
    import heartkit as hk

    task = hk.TaskFactory.get("arrhythmia")
    task.export(hk.HKExportParams(
        job_dir=Path("./results/arrhythmia-class-2"),
        ds_path=Path("./datasets"),
        datasets=[{
            "name": "icentia11k",
            "params": {}
        }],
        num_classes=2,
        class_map={
            0: 0,
            1: 1,
            2: 1
        },
        class_names=[
            "NONE", "AFIB/AFL"
        ],
        sampling_rate=200,
        frame_size=800,
        test_samples_per_patient=[100, 500, 100],
        model_file=Path("./results/arrhythmia-class-2/model.keras"),
        quantization={
            enabled=True,
            qat=False,
            ptq=True,
            input_type="int8",
            output_type="int8",
        },
        threshold=0.95,
        tflm_var_name="g_arrhythmia_model",
        tflm_file=Path("./results/arrhythmia-class-2/arrhythmia_model_buffer.h")
    ))
    ```

---

## [Demo](../modes/demo.md)

The `demo` command is used to run a task-level demonstration using either the PC or EVB as backend inference engine.

!!! Example "Python"

    The following snippet will run a task-level demo using the PC as the backend inference engine:

    ```python
    from pathlib import Path
    import heartkit as hk

    task = hk.TaskFactory.get("arrhythmia")

    task.demo(hk.HKDemoParams(
        job_dir=Path("./results/arrhythmia-class-2"),
        ds_path=Path("./datasets"),
        datasets=[{
            "name": "icentia11k",
            "params": {}
        }],
        num_classes=2,
        class_map={
            0: 0,
            1: 1,
            2: 1
        },
        class_names=[
            "NONE", "AFIB/AFL"
        ],
        sampling_rate=200,
        frame_size=800,
        model_file=Path("./results/arrhythmia-class-2/model.tflite"),
        backend="pc"
    ))
    ```

---
