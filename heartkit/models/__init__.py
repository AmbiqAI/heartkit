"""
# :material-graph: Models API

heartKIT provides a number of model architectures that can be used for training __heart-monitoring tasks__.
While a number of off-the-shelf models exist, they are often not efficient nor optimized for real-time, edge applications.
To address this, heartKIT provides a model factory that allows you to easily create and train customized models via [helia-edge](https://ambiqai.github.io/helia-edge/).
`helia-edge` includes a growing number of state-of-the-art models that can be easily configured and trained using high-level parameters.
The models are designed to be efficient and well-suited for real-time, edge applications.
Most of the models are based on state-of-the-art architectures that have been modified to allow for more fine-grain customization.
In addition, the models support 1D variants to allow for training on time-series data.

Please check [helia-edge](https://ambiqai.github.io/helia-edge/) for list of available models and their configurations.

## Available Models

- **[TCN](https://ambiqai.github.io/helia-edge/api/helia_edge/models/tcn)**: A CNN leveraging dilated convolutions (key=`tcn`)
- **[U-Net](https://ambiqai.github.io/helia-edge/api/helia_edge/models/unet)**: A CNN with encoder-decoder architecture for segmentation tasks (key=`unet`)
- **[U-NeXt](https://ambiqai.github.io/helia-edge/api/helia_edge/models/unext)**: A U-Net variant leveraging MBConv blocks (key=`unext`)
- **[EfficientNetV2](https://ambiqai.github.io/helia-edge/api/helia_edge/models/efficientnet)**: A CNN leveraging MBConv blocks (key=`efficientnet`)
- **[MobileOne](https://ambiqai.github.io/helia-edge/api/helia_edge/models/mobileone)**: A CNN aimed at sub-1ms inference (key=`mobileone`)
- **[ResNet](https://ambiqai.github.io/helia-edge/api/helia_edge/models/resnet)**: A popular CNN often used for vision tasks (key=`resnet`)
- **[Conformer](https://ambiqai.github.io/helia-edge/api/helia_edge/models/conformer)**: A transformer composed of both convolutional and self-attention blocks (key=`conformer`)
- **[MetaFormer](https://ambiqai.github.io/helia-edge/api/helia_edge/models/metaformer)**: A transformer composed of both spatial mixing and channel mixing blocks (key=`metaformer`)
- **[TSMixer](https://ambiqai.github.io/helia-edge/api/helia_edge/models/tsmixer)**: An All-MLP Architecture for Time Series Classification (key=`tsmixer`)


## Model Factory

The ModelFactory provides a convenient way to access the built-in models.

```py linenums="1"
import heartkit as hk

for model in hk.ModelFactory.list():
    print(f"Model name: {model} - {hk.ModelFactory.get(model)}")
```

## Usage

A model architecture can easily be instantied by providng a custom set of parameters to the model factory. Each model exposes a set of parameters defined using `Pydantic` to ensure type safety and consistency.


!!! Example

    The following example demonstrates how to create a TCN model using the `Tcn` class. The model is defined using a set of parameters defined in the `TcnParams` and `TcnBlockParams` classes.

    ```py linenums="1"
    import keras
    from helia_edge.models import TcnModel, TcnParams, TcnBlockParams

    inputs = keras.Input(shape=(800, 1))
    num_classes = 5

    model = TcnModel.model_from_params(
        x=inputs,
        params=TcnParams(
            input_kernel=(1, 3),
            input_norm="batch",
            blocks=[
                TcnBlockParams(filters=8, kernel=(1, 3), dilation=(1, 1), dropout=0.1, ex_ratio=1, se_ratio=0, norm="batch"),
                TcnBlockParams(filters=16, kernel=(1, 3), dilation=(1, 2), dropout=0.1, ex_ratio=1, se_ratio=0, norm="batch"),
                TcnBlockParams(filters=24, kernel=(1, 3), dilation=(1, 4), dropout=0.1, ex_ratio=1, se_ratio=4, norm="batch"),
                TcnBlockParams(filters=32, kernel=(1, 3), dilation=(1, 8), dropout=0.1, ex_ratio=1, se_ratio=4, norm="batch"),
            ],
            output_kernel=(1, 3),
            include_top=True,
            use_logits=True,
            model_name="tcn",
        ),
        num_classes=num_classes,
    )
    ```

"""

from typing import Protocol

import keras
import helia_edge as helia


class ModelFactoryItem(Protocol):
    def __call__(self, inputs: keras.KerasTensor, params: dict, num_classes: int) -> keras.Model: ...

    """ModelFactoryItem is a protocol for model factory items.

    Args:
        inputs (keras.KerasTensor): Input tensor
        params (dict): Model parameters
        num_classes (int): Number of classes

    Returns:
        keras.Model: Model
    """


ModelFactory = helia.utils.ItemFactory[ModelFactoryItem].shared("HKModelFactory")

ModelFactory.register("unet", helia.models.UNetModel.model_from_params)
ModelFactory.register("unext", helia.models.UNextModel.model_from_params)
ModelFactory.register("resnet", helia.models.ResNetModel.model_from_params)
ModelFactory.register("efficientnetv2", helia.models.EfficientNetV2Model.model_from_params)
ModelFactory.register("mobileone", helia.models.MobileNetV1Model.model_from_params)
ModelFactory.register("tcn", helia.models.TcnModel.model_from_params)
ModelFactory.register("composer", helia.models.ComposerModel.model_from_params)
