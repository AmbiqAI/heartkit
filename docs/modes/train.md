# Model Training

## <span class="sk-h2-span">Introduction </span>

Each task provides a mode to train a model on the specified datasets. The training mode can be invoked either via CLI or within `heartkit` python package. At a high level, the training mode performs the following actions based on the provided configuration parameters:

1. Load the configuration data (e.g. `rhythm-class-2.json`)
1. Load the desired datasets (e.g. `icentia11k`)
1. Load the custom model architecture (e.g. `tcn`)
1. Train the model
1. Save the trained model
1. Generate training report

---

## <span class="sk-h2-span">Usage</span>

!!! Example

    The following command will train a rhythm model using the reference configuration:

    === "CLI"

        ```bash
        heartkit --task rhythm --mode train --config ./configs/rhythm-class-2.json
        ```

    === "Python"

        --8<-- "assets/modes/python-train-snippet.md"

---

## <span class="sk-h2-span">Arguments </span>

The following tables lists the parameters that can be used to configure the training mode.


--8<-- "assets/modes/train-params.md"

---

## <span class="sk-h2-span">Logging</span>

__HeartKit__ provides built-in support for logging to several third-party services including [Weights & Biases](https://wandb.ai/site) (WANDB) and [TensorBoard](https://www.tensorflow.org/tensorboard).

### WANDB

The training mode is able to log all metrics and artifacts (aka models) to [Weights & Biases](https://wandb.ai/site) (WANDB). To enable WANDB logging, simply set environment variable `WANDB=1`. Remember to sign in prior to running experiments by running `wandb login`.


### TensorBoard

The training mode is able to log all metrics to [TensorBoard](https://www.tensorflow.org/tensorboard). To enable TensorBoard logging, simply set environment variable `TENSORBOARD=1`.
