# Task-Level Model Demo

## <span class="sk-h2-span">Introduction </span>

Each task in HeartKit has a corresponding demo mode that allows you to run a task-level demonstration using the specified backend inference engine (e.g. PC or EVB). This is useful to showcase the model's performance in real-time and to verify its accuracy in a real-world scenario. Similar to other modes, the demo can be invoked either via CLI or within `heartkit` python package. At a high level, the demo mode performs the following actions based on the provided configuration parameters:

1. Load the configuration file (e.g. `segmentation-class-2`)
1. Load the desired dataset features (e.g. `icentia11k`)
1. Load the trained model (e.g. `model.keras`)
1. Load random test subject's data
1. Perform inference via backend engine (e.g. PC or EVB)
1. Generate report

---

## <span class="sk-h2-span">Inference Backends</span>

HeartKit includes two built-in backend inference engines: PC and EVB. Additional backends can be easily added to the HeartKit framework by creating a new backend class and registering it to the backend factory.

### PC Backend

The PC backend is used to run the task-level demo on the local machine. This is useful for quick testing and debugging of the model.

1. Create / modify configuration file (e.g. `segmentation-class-2.json`)
1. Ensure "pc" is selected as the backend in configuration file.
1. Run demo `heartkit --mode demo --task segmentation --config ./configs/segmentation-class-2.json`
1. HTML report will be saved to `${job_dir}/report.html`

### EVB Backend

The EVB backend is used to run the task-level demo on an Ambiq EVB. This is useful to showcase the model's performance in real-time and to verify its accuracy in a real-world scenario.

1. Create / modify configuration file (e.g. `segmentation-class-2.json`)
1. Ensure "evb" is selected as the backend in configuration file.
1. Plug EVB into PC via two USB-C cables.
1. Build and flash firmware to EVB `cd evb && make && make deploy`
1. Run demo `heartkit --mode demo --task beat --config ./configs/segmentation-class-2.json`
1. HTML report will be saved to `${job_dir}/report.html`

### Bring-Your-Own-Backend

Similar to datasets, tasks, and models, the demo mode can be customized to use your own backend inference engine. HeartKit includes a backend factory (`BackendFactory`) that is used to create and run the backend engine.

#### How it Works

1. **Create a Backend**: Define a new backend by creating a new Python file. The file should contain a class that inherits from the `DemoBackend` base class and implements the required methods.

    ```python
    import heartkit as hk

    class CustomBackend(hk.HKBackend):
        def __init__(self, config):
            super().__init__(config)

        def run(self, model, data):
            pass
    ```

2. **Register the Backend**: Register the new backend with the `BackendFactory` by calling the `register` method. This method takes the backend name and the backend class as arguments.

    ```python
    import heartkit as hk
    hk.BackendFactory.register("custom", CustomBackend)
    ```

3. **Use the Backend**: The new backend can now be used by setting the `backend` flag in the demo configuration settings.

    ```python
    import heartkit as hk
    task = hk.TaskFactory.get("rhythm")
    task.demo(hk.HKDemoParams(
        ...,
        backend="custom"
    ))
    ```
    _OR_ by creating the backend directly:

    ```python
    import heartkit as hk
    backend = hk.BackendFactory.create("custom", config)
    ```

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
--8<-- "assets/tasks/segmentation/segmentation-demo.html"
</div>

---

## <span class="sk-h2-span">Arguments </span>

The following table lists the parameters that can be used to configure the demo mode. For argument `model_file`, the supported formats include `.keras` and `.tflite`.

### HKDemoParams

| Argument | Type | Opt/Req | Default | Description |
| --- | --- | --- | --- | --- |
| `job_dir` | `Path` | Optional | `./results` | Job output directory |
| `ds_path` | `Path` | Optional | `./datasets` | Dataset directory |
| `datasets` | `list[DatasetParams]` | Optional | `[]` | Datasets |
| `sampling_rate` | `int` | Optional | 250 | Target sampling rate (Hz) |
| `frame_size` | `int` | Optional | 1250 | Frame size |
| `num_classes` | `int` | Optional | 1 | # of classes |
| `class_map` | `dict[int, int]` | Optional | `{1: 1}` | Class/label mapping |
| `class_names` | `list[str]` | Optional | `None` | Class names |
| `preprocesses` | `list[PreprocessParams]` | Optional | `[]` | Preprocesses |
| `augmentations` | `list[AugmentationParams]` | Optional | `[]` | Augmentations |
| `model_file` | `Path` | Optional | `None` | Path to save model file (.keras) |
| `backend` | `Literal["pc", "evb"]` | Optional | `pc` | Backend |
| `demo_size` | `int` | Optional | `1000` | # samples for demo |
| `display_report` | `bool` | Optional | `True` | Display report |
| `seed` | `int` | Optional | `None` | Random state seed |
| `model_config` | `ConfigDict` | Optional | `{}` | Model configuration |

### QuantizationParams:

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


---
