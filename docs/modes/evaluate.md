# Model Evaluation

## <span class="sk-h2-span">Introduction </span>

Evaluate mode is used to test the performance of the model on the reserved test set for the specified task. Similar to training, the routine can be customized via CLI configuration file or by setting the parameters directly in the code. The evaluation process involves testing the model's performance on the test data to measure its accuracy, precision, recall, and F1 score. A number of results and metrics will be generated and saved to the `job_dir`.

---

## <span class="sk-h2-span">Usage</span>

!!! Example

    The following command will evaluate the rhythm model using the reference configuration:

    === "CLI"

        ```bash
        heartkit --mode evaluate --task rhythm --config ./configs/rhythm-class-2.json
        ```

    === "Python"

        ```python
        from pathlib import Path
        import heartkit as hk

        task = hk.TaskFactory.get("rhythm")
        task.evaluate(hk.HKTestParams(
            job_dir=Path("./results/rhythm-class-2"),
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
            data_parallelism=lambda: os.cpu_count() or 1,
            preprocesses=[
                hk.PreprocessParams(
                    name="znorm",
                    params=dict(
                        eps=0.01,
                        axis=None
                    )
                )
            ]
        ))
        ```

---

## <span class="sk-h2-span">Arguments </span>

The following tables lists the arguments that can be used with the `evaluate` command.

### `HKTestParams`

| Argument | Type | Opt/Req | Default | Description |
| --- | --- | --- | --- | --- |
| job_dir | Path | Optional | `tempfile.gettempdir` | Job output directory |
| ds_path | Path | Optional | `Path()` | Dataset directory |
| datasets | list[DatasetParams] | Optional |  | Datasets |
| sampling_rate | int | Optional | 250 | Target sampling rate (Hz) |
| frame_size | int | Optional | 1250 | Frame size |
| num_classes | int | Optional | 1 | # of classes |
| class_map | dict[int, int] | Optional |  | Class/label mapping |
| class_names | list[str] | Optional | None | Class names |
| test_samples_per_patient | int\|list[int] | Optional | 1000 | # test samples per patient |
| test_patients | float\|None | Optional | None | # or proportion of patients for testing |
| test_size | int | Optional | 200000 | # samples for testing |
| test_file | Path\|None | Optional | None | Path to load/store pickled test file |
| preprocesses | list[PreprocessParams] | Optional |  | Preprocesses |
| augmentations | list[AugmentationParams] | Optional |  | Augmentations |
| model_file | Path\|None | Optional | None | Path to save model file (.keras) |
| threshold | float\|None | Optional | None | Model output threshold |
| seed | int\|None | Optional | None | Random state seed |
| data_parallelism | int | Optional | `os.cpu_count` | # of data loaders running in parallel |

### QuantizationParams

| Argument | Type | Opt/Req | Default | Description |
| --- | --- | --- | --- | --- |
| enabled | bool | Optional | False | Enable quantization |
| qat | bool | Optional | False | Enable quantization aware training (QAT) |
| ptq | bool | Optional | False | Enable post training quantization (PTQ) |
| input_type | str\|None | Optional | None | Input type |
| output_type | str\|None | Optional | None | Output type |
| supported_ops | list[str]\|None | Optional | None | Supported ops |

### DatasetParams

| Argument | Type | Opt/Req | Default | Description |
| --- | --- | --- | --- | --- |
| name | str | Required |  | Dataset name |
| params | dict[str, Any] | Optional | {} | Dataset parameters |
| weight | float | Optional | 1 | Dataset weight |

### PreprocessParams

| Argument | Type | Opt/Req | Default | Description |
| --- | --- | --- | --- | --- |
| name | str | Required |  | Preprocess name |
| params | dict[str, Any] | Optional | {} | Preprocess parameters |

### AugmentationParams

| Argument | Type | Opt/Req | Default | Description |
| --- | --- | --- | --- | --- |
| name | str | Required |  | Augmentation name |
| params | dict[str, Any] | Optional | {} | Augmentation parameters |
