### HKTrainParams

| Argument | Type | Opt/Req | Default | Description |
| --- | --- | --- | --- | --- |
| name | str | Optional | "experiment" | Experiment name |
| job_dir | Path | Optional | `tempfile.gettempdir` | Job output directory |
| ds_path | Path | Optional | `Path()` | Dataset directory |
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
| architecture | ModelArchitecture\|None | Optional | None | Custom model architecture |
| model_file | Path\|None | Optional | None | Path to save model file (.keras) |
| weights_file | Path\|None | Optional | None | Path to a checkpoint weights to load |
| quantization | QuantizationParams | Optional |  | Quantization parameters |
| lr_rate | float | Optional | 1e-3 | Learning rate |
| lr_cycles | int | Optional | 3 | Number of learning rate cycles |
| lr_decay | float | Optional | 0.9 | Learning rate decay |
| class_weights | Literal["balanced", "fixed"] | Optional | "fixed" | Class weights |
| batch_size | int | Optional | 32 | Batch size |
| buffer_size | int | Optional | 100 | Buffer size |
| epochs | int | Optional | 50 | Number of epochs |
| steps_per_epoch | int | Optional | 10 | Number of steps per epoch |
| val_metric | Literal["loss", "acc", "f1"] | Optional | "loss" | Performance metric |
| preprocesses | list[PreprocessParams] | Optional | [] | Preprocesses |
| augmentations | list[AugmentationParams] | Optional | [] | Augmentations |
| seed | int\|None | Optional | None | Random state seed |
| data_parallelism | int | Optional | `os.cpu_count() or 1` | # of data loaders running in parallel |

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
