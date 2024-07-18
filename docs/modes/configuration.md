# Configuration Parameters

For each mode, a set of parameters are required to run the task. The following sections provide details on the parameters required for each mode.

### QuantizationParams

| Argument | Type | Opt/Req | Default | Description |
| --- | --- | --- | --- | --- |
| enabled | bool | Optional | False | Enable quantization |
| qat | bool | Optional | False | Enable quantization aware training (QAT) |
| format | QuantizationType | Optional | INT8 | Quantization mode |
| io_type | str | Optional | int8 | I/O type |
| conversion | ConversionType | Optional | KERAS | Conversion method |
| debug | bool | Optional | False | Debug quantization |
| fallback | bool | Optional | False | Fallback to float32 |


### ModelArchitecture

| Argument | Type | Opt/Req | Default | Description |
| --- | --- | --- | --- | --- |
| name | str | Required |  | Model architecture name |
| params | dict[str, Any] | Optional | {} | Model architecture parameters |

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


### DatasetParams

| Argument | Type | Opt/Req | Default | Description |
| --- | --- | --- | --- | --- |
| name | str | Required |  | Dataset name |
| path | Path | Optional | Path() | Dataset path |
| params | dict[str, Any] | Optional | {} | Parameters |
| weight | float | Optional | 1 | Dataset weight |


### HKDownloadParams

| Argument | Type | Opt/Req | Default | Description |
| --- | --- | --- | --- | --- |
| job_dir | Path | Optional | `tempfile.gettempdir` | Job output directory |
| datasets | list[DatasetParams] | Optional |  | Datasets |
| progress | bool | Optional | True | Display progress bar |
| force | bool | Optional | False | Force download dataset- overriding existing files |
| data_parallelism | int | Optional | `os.cpu_count` | # of data loaders running in parallel |

### HKTrainParams

| Argument | Type | Opt/Req | Default | Description |
| --- | --- | --- | --- | --- |
| name | str | Required | experiment | Experiment name |
| project | str | Required | heartkit | Project name |
| job_dir | Path | Optional | `tempfile.gettempdir` | Job output directory |
| datasets | list[DatasetParams] | Optional |  | Datasets |
| sampling_rate | int | Optional | 250 | Target sampling rate (Hz) |
| frame_size | int | Optional | 1250 | Frame size |
| num_classes | int | Optional | 1 | # of classes |
| class_map | dict[int, int] | Optional |  | Class/label mapping |
| class_names | list[str] | Optional | None | Class names |
| samples_per_patient | int\|list[int] | Optional | 1000 | # train samples per patient |
| val_samples_per_patient | int\|list[int] | Optional | 1000 | # validation samples per patient |
| train_patients | float\|None | Optional | None | # or proportion of patients for training |
| val_patients | float\|None | Optional | None | # or proportion of patients for validation |
| val_file | Path\|None | Optional | None | Path to load/store pickled validation file |
| val_size | int\|None | Optional | None | # samples for validation |
| resume | bool | Optional | False | Resume training |
| architecture | ModelArchitecture | Optional |  | Custom model architecture |
| model_file | Path\|None | Optional | None | Path to save model file (.keras) |
| threshold | float\|None | Optional | None | Model output threshold |
| weights_file | Path\|None | Optional | None | Path to a checkpoint weights to load |
| quantization | QuantizationParams | Optional |  | Quantization parameters |
| lr_rate | float | Optional | 0.001 | Learning rate |
| lr_cycles | int | Optional | 3 | Number of learning rate cycles |
| lr_decay | float | Optional | 0.9 | Learning rate decay |
| class_weights | Literal["balanced", "fixed"] | Optional | fixed | Class weights |
| label_smoothing | float | Optional | 0 | Label smoothing |
| batch_size | int | Optional | 32 | Batch size |
| buffer_size | int | Optional | 100 | Buffer size |
| epochs | int | Optional | 50 | Number of epochs |
| steps_per_epoch | int | Optional | 10 | Number of steps per epoch |
| val_metric | Literal["loss", "acc", "f1"] | Optional | loss | Performance metric |
| preprocesses | list[PreprocessParams] | Optional |  | Preprocesses |
| augmentations | list[AugmentationParams] | Optional |  | Augmentations |
| seed | int\|None | Optional | None | Random state seed |
| data_parallelism | int | Optional | `os.cpu_count` | # of data loaders running in parallel |
| verbose | int | Optional | 1 | Verbosity level |

### HKTestParams

| Argument | Type | Opt/Req | Default | Description |
| --- | --- | --- | --- | --- |
| name | str | Required | experiment | Experiment name |
| project | str | Required | heartkit | Project name |
| job_dir | Path | Optional | `tempfile.gettempdir` | Job output directory |
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
| verbose | int | Optional | 1 | Verbosity level |

### HKExportParams

| Argument | Type | Opt/Req | Default | Description |
| --- | --- | --- | --- | --- |
| name | str | Required | experiment | Experiment name |
| project | str | Required | heartkit | Project name |
| job_dir | Path | Optional | `tempfile.gettempdir` | Job output directory |
| datasets | list[DatasetParams] | Optional |  | Datasets |
| sampling_rate | int | Optional | 250 | Target sampling rate (Hz) |
| frame_size | int | Optional | 1250 | Frame size |
| num_classes | int | Optional | 3 | # of classes |
| class_map | dict[int, int] | Optional |  | Class/label mapping |
| class_names | list[str] | Optional | None | Class names |
| test_samples_per_patient | int\|list[int] | Optional | 100 | # test samples per patient |
| test_patients | float\|None | Optional | None | # or proportion of patients for testing |
| test_size | int | Optional | 100000 | # samples for testing |
| test_file | Path\|None | Optional | None | Path to load/store pickled test file |
| preprocesses | list[PreprocessParams] | Optional |  | Preprocesses |
| augmentations | list[AugmentationParams] | Optional |  | Augmentations |
| model_file | Path\|None | Optional | None | Path to save model file (.keras) |
| threshold | float\|None | Optional | None | Model output threshold |
| val_acc_threshold | float\|None | Optional | 0.98 | Validation accuracy threshold |
| use_logits | bool | Optional | True | Use logits output or softmax |
| quantization | QuantizationParams | Optional |  | Quantization parameters |
| tflm_var_name | str | Optional | g_model | TFLite Micro C variable name |
| tflm_file | Path\|None | Optional | None | Path to copy TFLM header file (e.g. ./model_buffer.h) |
| data_parallelism | int | Optional | `os.cpu_count` | # of data loaders running in parallel |
| model_config | ConfigDict | Optional |  | Model configuration |
| verbose | int | Optional | 1 | Verbosity level |

### HKDemoParams

| Argument | Type | Opt/Req | Default | Description |
| --- | --- | --- | --- | --- |
| name | str | Required | experiment | Experiment name |
| project | str | Required | heartkit | Project name |
| job_dir | Path | Optional | `tempfile.gettempdir` | Job output directory |
| datasets | list[DatasetParams] | Optional |  | Datasets |
| sampling_rate | int | Optional | 250 | Target sampling rate (Hz) |
| frame_size | int | Optional | 1250 | Frame size |
| num_classes | int | Optional | 1 | # of classes |
| class_map | dict[int, int] | Optional |  | Class/label mapping |
| class_names | list[str] | Optional | None | Class names |
| preprocesses | list[PreprocessParams] | Optional |  | Preprocesses |
| augmentations | list[AugmentationParams] | Optional |  | Augmentations |
| model_file | Path\|None | Optional | None | Path to save model file (.keras) |
| backend | str | Optional | pc | Backend |
| demo_size | int | Optional | 1000 | # samples for demo |
| display_report | bool | Optional | True | Display report |
| seed | int\|None | Optional | None | Random state seed |
| model_config | ConfigDict | Optional |  | Model configuration |
| verbose | int | Optional | 1 | Verbosity level |
