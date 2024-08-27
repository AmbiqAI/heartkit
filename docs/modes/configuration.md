# :material-code-json: Configuration Parameters

For each mode, common configuration parameters, [HKTaskParams](#hktaskparams), are required to run the task. These parameters are used to define the task, datasets, model, and other settings. Rather than defining separate configuration files for each mode, a single configuration object is used to simplify configuration files and heavy re-use of parameters between modes.

## <span class="sk-h2-span">QuantizationParams</span>

Quantization parameters define the quantization-aware training (QAT) and post-training quantization (PTQ) settings. This is used for modes: train, evaluate, export, and demo.

| Argument | Type | Opt/Req | Default | Description |
| --- | --- | --- | --- | --- |
| enabled | bool | Optional | False | Enable quantization |
| qat | bool | Optional | False | Enable quantization aware training (QAT) |
| format | Literal["int8", "int16", "float16"] | Optional | int8 | Quantization mode |
| io_type | str | Optional | int8 | I/O type |
| conversion | Literal["keras", "tflite"] | Optional | keras | Conversion method |
| debug | bool | Optional | False | Debug quantization |
| fallback | bool | Optional | False | Fallback to float32 |

## <span class="sk-h2-span">NamedParams</span>

Named parameters are used to provide custom parameters for a given object or callable where parameter types are not known ahead of time. For example, a dataset, 'CustomDataset', may require custom parameters such as 'path', 'label', 'sampling_rate', etc. When a task loads the dataset using `name`, the task will then unpack the custom parameters and pass them to the dataset initializer.

| Argument | Type | Opt/Req | Default | Description |
| --- | --- | --- | --- | --- |
| name | str | Required |  | Named parameters name |
| params | dict[str, Any] | Optional | {} | Named parameters |

```py linenums="1"

import heartkit as hk

class CustomDataset(hk.HKDataset):

    def __init__(self, a: int = 1, b: int = 2) -> None:
        self.a = a
        self.b = b

hk.DatasetFactory.register("custom", CustomDataset)

params = hk.HKTaskParams(
    datasets=[
        hk.NamedParams(
            name="custom",
            params=dict(a=1, b=2)
        )
    ]
)

```

## <span class="sk-h2-span">HKTaskParams</span>

These parameters are supplied to a [Task](../tasks/index.md) when running a given mode such as `train`, `evaluate`, `export`, or `demo`. A single configuration object is used to simplify configuration files and heavy re-use of parameters between modes.


| Argument | Type | Opt/Req | Default | Description | Mode |
| --- | --- | --- | --- | --- | --- |
| name | str | Required | experiment | Experiment name | All |
| project | str | Required | heartkit | Project name | All |
| job_dir | Path | Optional | `tempfile.gettempdir` | Job output directory | All |
| datasets | list[NamedParams] | Optional |  | Datasets | All |
| force_download | bool | Optional | False | Force download datasets | download |
| dataset_weights | list[float]\|None | Optional | None | Dataset weights | train |
| sampling_rate | int | Optional | 250 | Target sampling rate (Hz) | All |
| frame_size | int | Optional | 1250 | Frame size in samples | All |
| samples_per_patient | int\|list[int] | Optional | 1000 | # train samples per patient | train |
| val_samples_per_patient | int\|list[int] | Optional | 1000 | # validation samples per patient | train |
| test_samples_per_patient | int\|list[int] | Optional | 1000 | # test samples per patient | evaluate |
| train_patients | float\|None | Optional | None | # or proportion of patients for training | train |
| val_patients | float\|None | Optional | None | # or proportion of patients for validation | train |
| test_patients | float\|None | Optional | None | # or proportion of patients for testing | evaluate |
| val_file | Path\|None | Optional | None | Path to load/store pickled validation file | train |
| test_file | Path\|None | Optional | None | Path to load/store pickled test file | evaluate, export |
| val_size | int\|None | Optional | None | # samples for validation | train |
| test_size | int | Optional | 10000 | # samples for testing | evaluate, export |
| num_classes | int | Optional | 1 | # of classes | All |
| class_map | dict[int, int] | Optional |  | Class/label mapping | All |
| class_names | list[str]\|None | Optional | None | Class names | All |
| resume | bool | Optional | False | Resume training | train |
| architecture | NamedParams\|None | Optional | None | Custom model architecture | train |
| model_file | Path\|None | Optional | None | Path to load/save model file (.keras) | All |
| use_logits | bool | Optional | True | Use logits output or softmax | Export |
| weights_file | Path\|None | Optional | None | Path to a checkpoint weights to load/save | train |
| quantization | QuantizationParams | Optional |  | Quantization parameters | All |
| lr_rate | float | Optional | 0.001 | Learning rate | train |
| lr_cycles | int | Optional | 3 | Number of learning rate cycles | train |
| lr_decay | float | Optional | 0.9 | Learning rate decay | train |
| label_smoothing | float | Optional | 0 | Label smoothing | train |
| batch_size | int | Optional | 32 | Batch size | train |
| buffer_size | int | Optional | 100 | Buffer cache size | train |
| epochs | int | Optional | 50 | Number of epochs | train |
| steps_per_epoch | int | Optional | 10 | Number of steps per epoch | train |
| val_steps_per_epoch | int | Optional | 10 | Number of validation steps | train |
| val_metric | Literal["loss", "acc", "f1"] | Optional | loss | Performance metric | train |
| class_weights | Literal["balanced", "fixed"] | Optional | fixed | Class weights | train |
| threshold | float\|None | Optional | None | Model output threshold | evaluate, export |
| val_metric_threshold | float\|None | Optional | 0.98 | Validation metric threshold | export |
| test_metric_threshold | float\|None | Optional | 0.98 | Test metric threshold | export |
| tflm_var_name | str | Optional | g_model | TFLite Micro C variable name | export |
| tflm_file | Path\|None | Optional | None | Path to copy TFLM header file (e.g. ./model_buffer.h) | export |
| backend | str | Optional | pc | Backend | demo |
| demo_size | int\|None | Optional | 1000 | # samples for demo | demo |
| display_report | bool | Optional | True | Display report | demo |
| seed | int\|None | Optional | None | Random state seed | All |
| data_parallelism | int | Optional | `os.cpu_count` | # of data loaders running in parallel | All |
| verbose | int | Optional | 1 | Verbosity level | All |
