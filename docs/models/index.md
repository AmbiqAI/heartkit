# :material-graph-outline: Models

HeartKit provides a number of model architectures that can be used for training __heart-monitoring tasks__. While a number of off-the-shelf models exist, they are often not efficient nor optimized for real-time, edge applications. To address this, HeartKit provides a model factory that allows you to easily create and train customized models via [neuralspot-edge](https://ambiqai.github.io/neuralspot-edge/). `neuralspot-edge` includes a growing number of state-of-the-art models that can be easily configured and trained using high-level parameters. The models are designed to be efficient and well-suited for real-time, edge applications. Most of the models are based on state-of-the-art architectures that have been modified to allow for more fine-grain customization. In addition, the models support 1D variants to allow for training on time-series data. Please check [neuralspot-edge](https://ambiqai.github.io/neuralspot-edge/) for list of available models and their configurations.

---

## <span class="sk-h2-span">Available Models</span>

- **[TCN](https://ambiqai.github.io/neuralspot-edge/models/tcn)**: A CNN leveraging dilated convolutions (key=`tcn`)
- **[U-Net](https://ambiqai.github.io/neuralspot-edge/models/unet)**: A CNN with encoder-decoder architecture for segmentation tasks (key=`unet`)
- **[U-NeXt](https://ambiqai.github.io/neuralspot-edge/models/unext)**: A U-Net variant leveraging MBConv blocks (key=`unext`)
- **[EfficientNetV2](https://ambiqai.github.io/neuralspot-edge/models/efficientnet)**: A CNN leveraging MBConv blocks (key=`efficientnet`)
- **[MobileOne](https://ambiqai.github.io/neuralspot-edge/models/mobileone)**: A CNN aimed at sub-1ms inference (key=`mobileone`)
- **[ResNet](https://ambiqai.github.io/neuralspot-edge/models/resnet)**: A popular CNN often used for vision tasks (key=`resnet`)
- **[Conformer](https://ambiqai.github.io/neuralspot-edge/models/conformer)**: A transformer composed of both convolutional and self-attention blocks (key=`conformer`)
- **[MetaFormer](https://ambiqai.github.io/neuralspot-edge/models/metaformer)**: A transformer composed of both spatial mixing and channel mixing blocks (key=`metaformer`)
- **[TSMixer](https://ambiqai.github.io/neuralspot-edge/models/tsmixer)**: An All-MLP Architecture for Time Series Classification (key=`tsmixer`)
* **[Bring-Your-Own-Model](./byom.md)**: Add a custom model architecture to HeartKit.

---

## <span class="sk-h2-span">Model Factory</span>

HeartKit includes a model factory, `ModelFactory`, that eases the processes of creating models for training. The factory allows you to create models by specifying the model key and the model parameters. The factory will then create the model using the specified parameters. The factory also allows you to register custom models that can be used for training. By leveraring a factory, a task only needs to provide the architecture key and the parameters, and the factory will take care of the rest.

The model factory provides the following methods:

* **hk.ModelFactory.register**: Register a custom model
* **hk.ModelFactory.unregister**: Unregister a custom model
* **hk.ModelFactory.has**: Check if a model is registered
* **hk.ModelFactory.get**: Get a model from the factory
* **hk.ModelFactory.list**: List all available models

---

## <span class="sk-h2-span">Usage</span>

### Defining a model in configuration file

A model can be created when invoking a command via the CLI by setting [architecture](../modes/configuration.md#hktaskparams) in the configuration file. The task will use the supplied name to get the registered model and instantiate with the provided parameters.

Given the following configuration file `configuration.json`:

```json
{
    ...
    "architecture:" {
        "name": "tcn",
        "params": {
            "input_kernel": [1, 3],
            "input_norm": "batch",
            "blocks": [
                {"depth": 1, "branch": 1, "filters": 12, "kernel": [1, 3], "dilation": [1, 1], "dropout": 0.10, "ex_ratio": 1, "se_ratio": 0, "norm": "batch"},
                {"depth": 1, "branch": 1, "filters": 20, "kernel": [1, 3], "dilation": [1, 1], "dropout": 0.10, "ex_ratio": 1, "se_ratio": 2, "norm": "batch"},
                {"depth": 1, "branch": 1, "filters": 28, "kernel": [1, 3], "dilation": [1, 2], "dropout": 0.10, "ex_ratio": 1, "se_ratio": 2, "norm": "batch"},
                {"depth": 1, "branch": 1, "filters": 36, "kernel": [1, 3], "dilation": [1, 4], "dropout": 0.10, "ex_ratio": 1, "se_ratio": 2, "norm": "batch"},
                {"depth": 1, "branch": 1, "filters": 40, "kernel": [1, 3], "dilation": [1, 8], "dropout": 0.10, "ex_ratio": 1, "se_ratio": 2, "norm": "batch"}
            ],
            "output_kernel": [1, 3],
            "include_top": true,
            "use_logits": true,
            "model_name": "tcn"
        }
    }
}
```

### Defining a model in code

The model can be created using the following command:

```bash
heartkit --mode train --task rhythm --config config.json
```

Alternatively, the model can be created directly in code using the following snippet:

```py linenums="1"

import keras
import heartkit as hk

architecture = {
    "name": "tcn",
    "params": {
        "input_kernel": [1, 3],
        "input_norm": "batch",
        "blocks": [
            {"depth": 1, "branch": 1, "filters": 12, "kernel": [1, 3], "dilation": [1, 1], "dropout": 0.10, "ex_ratio": 1, "se_ratio": 0, "norm": "batch"},
            {"depth": 1, "branch": 1, "filters": 20, "kernel": [1, 3], "dilation": [1, 1], "dropout": 0.10, "ex_ratio": 1, "se_ratio": 2, "norm": "batch"},
            {"depth": 1, "branch": 1, "filters": 28, "kernel": [1, 3], "dilation": [1, 2], "dropout": 0.10, "ex_ratio": 1, "se_ratio": 2, "norm": "batch"},
            {"depth": 1, "branch": 1, "filters": 36, "kernel": [1, 3], "dilation": [1, 4], "dropout": 0.10, "ex_ratio": 1, "se_ratio": 2, "norm": "batch"},
            {"depth": 1, "branch": 1, "filters": 40, "kernel": [1, 3], "dilation": [1, 8], "dropout": 0.10, "ex_ratio": 1, "se_ratio": 2, "norm": "batch"}
        ],
        "output_kernel": [1, 3],
        "include_top": True,
        "use_logits": True,
        "model_name": "tcn"
    }
}

inputs = keras.Input(shape=(256,1), dtype="float32")
num_classes = 5

model = hk.ModelFactory.get(architecture["name"])(
    x=inputs,
    params=architecture["params"],
    num_classes=num_classes,
)

model.summary()
```

---
