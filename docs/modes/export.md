# Model Exporting

## <span class="sk-h2-span">Introduction </span>

Export mode is used to convert the trained TensorFlow model into both TensorFlow Lite (TFL) and TensorFlow Lite for micro-controller (TFLM) variants. The command will also verify the models' outputs match. Post-training quantization (PTQ) can also be enabled by setting the `quantization` flag in the configuration.


## <span class="sk-h2-span">Usage</span>

!!! Example

    The following command will export the 2-class arrhythmia model to TF Lite and TFLM:

    === "CLI"

        ```bash
        heartkit --mode export --task arrhythmia --config ./configs/arrhythmia-class-2.json
        ```

    === "Python"

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


## <span class="sk-h2-span">Arguments </span>

The following table lists the parameters that can be used to configure the export mode. Once converted, the TFLM header file will be copied to location specified by `tflm_file`. The `threshold` flag can be used to set the model's output threshold.  The `use_logits` flag can be used to set the model's output to use logits or softmax.

| Argument | Type | Opt/Req | Default | Description |
| --- | --- | --- | --- | --- |
| job_dir | Path | Optional | `tempfile.gettempdir` | Job output directory |
| ds_path | Path | Optional | `Path()` | Dataset directory |
| sampling_rate | int | Optional | 250 | Target sampling rate (Hz) |
| frame_size | int | Optional | 1250 | Frame size |
| num_classes | int | Optional | 3 | # of classes |
| preprocesses | list[PreprocessParams] | Optional |  | Preprocesses |
| test_samples_per_patient | int\|list[int] | Optional | 100 | # test samples per patient |
| test_patients | float\|None | Optional | None | # or proportion of patients for testing |
| test_size | int | Optional | 100000 | # samples for testing |
| model_file | str\|None | Optional | None | Path to model file |
| threshold | float\|None | Optional | None | Model output threshold |
| val_acc_threshold | float\|None | Optional | 0.98 | Validation accuracy threshold |
| use_logits | bool | Optional | True | Use logits output or softmax |
| quantization | bool\|None | Optional | None | Enable post training quantization (PQT) |
| tflm_var_name | str | Optional | "g_model" | TFLite Micro C variable name |
| tflm_file | Path\|None | Optional | None | Path to copy TFLM header file (e.g. ./model_buffer.h) |
| data_parallelism | int | Optional | `lambda: os.cpu_count() or 1` | # of data loaders running in parallel |
