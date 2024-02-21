# Model Evaluation

## <span class="sk-h2-span">Introduction </span>

Evaluate mode is used to test the performance of the model on the reserved test set for the specified task. For certain tasks, a confidence threshold can also be set such that a label is only assigned when the model's probability is greater than the threshold; otherwise, a label of inconclusive will be assigned.

---

## <span class="sk-h2-span">Usage</span>

!!! Example

    The following command will evaluate the 2-class arrhythmia model using the reference configuration:

    === "CLI"

        ```bash
        heartkit --mode evaluate --task arrhythmia --config ./configs/arrhythmia-class-2.json
        ```

    === "Python"

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

The following table lists the arguments that can be used with the `evaluate` command.

| Argument | Type | Opt/Req | Default | Description |
| --- | --- | --- | --- | --- |
| job_dir | Path | Optional | `tempfile.gettempdir` | Job output directory |
| ds_path | Path | Optional | `Path()` | Dataset directory |
| sampling_rate | int | Optional | 250 | Target sampling rate (Hz) |
| frame_size | int | Optional | 1250 | Frame size |
| num_classes | int | Optional | 3 | # of classes |
| test_samples_per_patient | int\|list[int] | Optional | 1000 | # test samples per patient |
| test_patients | float\|None | Optional | None | # or proportion of patients for testing |
| test_size | int | Optional | 200000 | # samples for testing |
| data_parallelism | int | Optional | `lambda: os.cpu_count() or 1` | # of data loaders running in parallel |
| preprocesses | list[PreprocessParams] | Optional |  | Preprocesses |
| model_file | str\|None | Optional | None | Path to model file |
| threshold | float\|None | Optional | None | Model output threshold |
| seed | int\|None | Optional | None | Random state seed |
