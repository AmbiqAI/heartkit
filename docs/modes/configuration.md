# Configuration Parameters

For each mode, a set of parameters are required to run the task. The following sections provide details on the parameters required for each mode.

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

Named parameters are used to provide custom parameters for a given object or callable. For example, a dataset, 'my-dataset', may require custom parameters such as 'path', 'label', 'sampling_rate', etc. When a task loads the dataset using `name`, the task will then unpack the custom parameters and pass them to the dataset loader.

| Argument | Type | Opt/Req | Default | Description |
| --- | --- | --- | --- | --- |
| name | str | Required |  | Named parameters name |
| params | dict[str, Any] | Optional | {} | Named parameters |

## <span class="sk-h2-span">HKTaskParams</span>

These parameters are supplied to a [Task](../tasks/index.md) when running a given mode such as `train`, `evaluate`, `export`, or `demo`. A single configuration object is used to simplify configuration files and heavy re-use of parameters between modes.


| Argument | Type | Opt/Req | Default | Description |
| --- | --- | --- | --- | --- |
| name | str | Required | experiment | Experiment name |
| project | str | Required | heartkit | Project name |
| job_dir | Path | Optional | `tempfile.gettempdir` | Job output directory |
| datasets | list[NamedParams] | Optional |  | Datasets |
| force_download | bool | Optional | False | Force download datasets |
| dataset_weights | list[float]\|None | Optional | None | Dataset weights |
| sampling_rate | int | Optional | 250 | Target sampling rate (Hz) |
| frame_size | int | Optional | 1250 | Frame size in samples |
| samples_per_patient | int\|list[int] | Optional | 1000 | # train samples per patient |
| val_samples_per_patient | int\|list[int] | Optional | 1000 | # validation samples per patient |
| test_samples_per_patient | int\|list[int] | Optional | 1000 | # test samples per patient |
| train_patients | float\|None | Optional | None | # or proportion of patients for training |
| val_patients | float\|None | Optional | None | # or proportion of patients for validation |
| test_patients | float\|None | Optional | None | # or proportion of patients for testing |
| val_file | Path\|None | Optional | None | Path to load/store pickled validation file |
| test_file | Path\|None | Optional | None | Path to load/store pickled test file |
| val_size | int\|None | Optional | None | # samples for validation |
| test_size | int | Optional | 10000 | # samples for testing |
| num_classes | int | Optional | 1 | # of classes |
| class_map | dict[int, int] | Optional |  | Class/label mapping |
| class_names | list[str]\|None | Optional | None | Class names |
| resume | bool | Optional | False | Resume training |
| architecture | NamedParams\|None | Optional | None | Custom model architecture |
| model_file | Path\|None | Optional | None | Path to load/save model file (.keras) |
| use_logits | bool | Optional | True | Use logits output or softmax |
| weights_file | Path\|None | Optional | None | Path to a checkpoint weights to load/save |
| quantization | QuantizationParams | Optional |  | Quantization parameters |
| lr_rate | float | Optional | 0.001 | Learning rate |
| lr_cycles | int | Optional | 3 | Number of learning rate cycles |
| lr_decay | float | Optional | 0.9 | Learning rate decay |
| label_smoothing | float | Optional | 0 | Label smoothing |
| batch_size | int | Optional | 32 | Batch size |
| buffer_size | int | Optional | 100 | Buffer cache size |
| epochs | int | Optional | 50 | Number of epochs |
| steps_per_epoch | int | Optional | 10 | Number of steps per epoch |
| val_steps_per_epoch | int | Optional | 10 | Number of validation steps |
| val_metric | Literal["loss", "acc", "f1"] | Optional | loss | Performance metric |
| class_weights | Literal["balanced", "fixed"] | Optional | fixed | Class weights |
| threshold | float\|None | Optional | None | Model output threshold |
| val_metric_threshold | float\|None | Optional | 0.98 | Validation metric threshold |
| tflm_var_name | str | Optional | g_model | TFLite Micro C variable name |
| tflm_file | Path\|None | Optional | None | Path to copy TFLM header file (e.g. ./model_buffer.h) |
| backend | str | Optional | pc | Backend |
| demo_size | int\|None | Optional | 1000 | # samples for demo |
| display_report | bool | Optional | True | Display report |
| seed | int\|None | Optional | None | Random state seed |
| data_parallelism | int | Optional | `os.cpu_count` | # of data loaders running in parallel |
| verbose | int | Optional | 1 | Verbosity level |
