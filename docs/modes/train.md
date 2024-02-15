# Model Training

## <span class="sk-h2-span">Introduction </span>

Each task provides a mode to train a model on the specified datasets. The training mode can be invoked either via CLI or within `heartkit` python package. At a high level, the training mode performs the following actions based on the provided configuration parameters:

1. Load the configuration data (e.g. `arrhythmia-class-2.json`)
1. Load the desired datasets (e.g. `icentia11k`)
1. Load the model architecture (e.g. `tcn`)
1. Train the model
1. Save the trained model
1. Generate training report

---

## <span class="sk-h2-span">Usage</span>

!!! Example

    The following command will train a 2-class arrhythmia model using the reference configuration:

    === "CLI"

        ```bash
        heartkit --task arrhythmia --mode train --config ./configs/arrhythmia-class-2.json
        ```

    === "Python"

        ```python
        from pathlib import Path
        import heartkit as hk

        task = hk.TaskFactory.get("arrhythmia")
        task.train(hk.HKTrainParams(
            job_dir=Path("./results/arrhythmia-class-2"),
            ds_path=Path("./datasets"),
            datasets=[{"name": "icentia11k", "params": {}}],
            sampling_rate=200,
            frame_size=800,
            num_classes=2,
            samples_per_patient=[100, 800],
            val_samples_per_patient=[100, 800],
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

## <span class="sk-h2-span">Arguments </span>

The following table lists the parameters that can be used to configure the training mode.

| Argument | Type | Opt/Req | Default | Description |
| --- | --- | --- | --- | --- |
| job_dir | Path | Optional | `tempfile.gettempdir` | Job output directory |
| ds_path | Path | Optional | `Path()` | Dataset directory |
| sampling_rate | int | Optional | 250 | Target sampling rate (Hz) |
| frame_size | int | Optional | 1250 | Frame size |
| num_classes | int | Optional | 3 | # of classes |
| samples_per_patient | int\|list[int] | Optional | 1000 | # train samples per patient |
| val_samples_per_patient | int\|list[int] | Optional | 1000 | # validation samples per patient |
| train_patients | float\|None | Optional | None | # or proportion of patients for training |
| val_patients | float\|None | Optional | None | # or proportion of patients for validation |
| val_file | Path\|None | Optional | None | Path to load/store pickled validation file |
| val_size | int\|None | Optional | None | # samples for validation |
| data_parallelism | int | Optional | `lambda: os.cpu_count() or 1` | # of data loaders running in parallel |
| model | str\|None | Optional | None | Custom model |
| model_file | str\|None | Optional | None | Path to model file |
| model_params | dict[str, Any]\|None | Optional | None | Custom model parameters |
| weights_file | Path\|None | Optional | None | Path to a checkpoint weights to load |
| quantization | bool\|None | Optional | None | Enable quantization aware training (QAT) |
| batch_size | int | Optional | 32 | Batch size |
| buffer_size | int | Optional | 100 | Buffer size |
| epochs | int | Optional | 50 | Number of epochs |
| steps_per_epoch | int\|None | Optional | None | Number of steps per epoch |
| val_metric | Literal\["loss", "acc", "f1"\] | Optional | "loss" | Performance metric |
| preprocesses | list[PreprocessParams] | Optional |  | Preprocesses |
| augmentations | list[AugmentationParams] | Optional |  | Augmentations |
| seed | int\|None | Optional | None | Random state seed |


## <span class="sk-h2-span">Logging</span>

__HeartKit__ provides built-in support for logging to several third-party services including [Weights & Biases](https://wandb.ai/site) (WANDB) and [TensorBoard](https://www.tensorflow.org/tensorboard).

### WANDB

The training mode is able to log all metrics and artifacts (aka models) to [Weights & Biases](https://wandb.ai/site) (WANDB). To enable WANDB logging, simply set environment variable `WANDB=1`. Remember to sign in prior to running experiments by running `wandb login`.


### TensorBoard

The training mode is able to log all metrics to [TensorBoard](https://www.tensorflow.org/tensorboard). To enable TensorBoard logging, simply set environment variable `TENSORBOARD=1`.
