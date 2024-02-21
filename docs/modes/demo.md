# Task-Level Model Demo

## <span class="sk-h2-span">Introduction </span>

Each task in HeartKit has a corresponding demo mode that allows you to run a task-level demonstration using either the PC or EVB as the backend inference engine. Similar to other modes, the demo can be invoked either via CLI or within `heartkit` python package. At a high level, the demo performs the following actions based on the provided configuration parameters:

1. Load the configuration file (e.g. `segmentation-class-2`)
1. Load the desired dataset features (e.g. `icentia11k`)
1. Load the trained model (e.g. `model.keras`)
1. Load random test subject's data
1. Perform inference either on PC or EVB
1. Generate report

---

## <span class="sk-h2-span">Backends</span>

### PC backend

1. Create / modify configuration file (e.g. `segmentation-class-2.json`)
1. Ensure "pc" is selected as the backend in configuration file.
1. Run demo `heartkit --mode demo --task segmentation --config ./configs/segmentation-class-2.json`
1. HTML report will be saved to `${job_dir}/report.html`

### EVB backend

1. Create / modify configuration file (e.g. `segmentation-class-2.json`)
1. Ensure "evb" is selected as the backend in configuration file.
1. Plug EVB into PC via two USB-C cables.
1. Build and flash firmware to EVB `cd evb && make && make deploy`
1. Run demo `heartkit --mode demo --task beat --config ./configs/segmentation-class-2.json`
1. HTML report will be saved to `${job_dir}/report.html`

---

## <span class="sk-h2-span">Arguments </span>

The following table lists the parameters that can be used to configure the demo mode. For argument `model_file`, the supported formats include `.keras` and `.tflite`.

| Argument | Type | Opt/Req | Default | Description |
| --- | --- | --- | --- | --- |
| job_dir | Path | Optional | `tempfile.gettempdir` | Job output directory |
| ds_path | Path | Optional | `Path()` | Dataset base directory |
| sampling_rate | int | Optional | 250 | Target sampling rate (Hz) |
| frame_size | int | Optional | 1250 | Frame size |
| num_classes | int | Optional | 3 | # of classes |
| preprocesses | list[PreprocessParams] | Optional |  | Preprocesses |
| model_file | str\|None | Optional | None | Path to model file |
| backend | Literal["pc", "evb"] | Optional | "pc" | Backend |
| seed | int\|None | Optional | None | Random state seed |

---

## <span class="sk-h2-span">Usage </span>

The following is an example of a task-level demo report for the segmentation task. Upon running segmentation, the demo will extract inter-beat-intervals (IBIs) and report various HR and HRV metrics. These metrics are computed using Ambiq's [PhysioKit Python Package](https://ambiqai.github.io/physiokit)- a toolkit to process raw ambulatory bio-signals.

=== "CLI"

    ```bash
    heartkit -m export -t segmentation -c ./configs/segmentation-class-2.json
    ```

=== "Python"

    ```python
    from pathlib import Path
    import heartkit as hk

    task = hk.TaskFactory.get("segmentation")
    task.export(hk.HKDemoParams(
        job_dir=Path("./results/segmentation-class-2"),
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
        sampling_rate=100,
        frame_size=256,
        backend="pc",
        model_file=Path("./results/segmentation-class-2/model.keras"),
    ))
    ```

<div class="sk-plotly-graph-div">
--8<-- "assets/segmentation-demo.html"
</div>

---
