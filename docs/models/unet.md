# U-Net

## <span class="sk-h2-span">Overview</span>

U-Net is a type of convolutional neural network (CNN) that is commonly used for segmentation tasks. U-Net is a fully convolutional network that consists of a series of convolutional layers and pooling layers. The pooling layers are used to downsample the input while the convolutional layers are used to upsample the input. The skip connections between the pooling layers and convolutional layers allow U-Net to preserve spatial/temporal information while also allowing for faster training and inference times.

For more info, refer to the original paper [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://doi.org/10.1007/978-3-319-24574-4_28).

---

## <span class="sk-h2-span">Additions</span>

The U-Net architecture has been modified to allow the following:

* Enable 1D and 2D variants.
* Convolutional pairs can factorized into depthwise separable convolutions.
* Specifiy the number of convolutional layers per block both downstream and upstream.
* Normalization can be set between batch normalization and layer normalization.
* ReLU is replaced with the approximated ReLU6.

---

## <span class="sk-h2-span">Arguments</span>

The following arguments can be passed to the `U-Net` class:

### UNetParams

| Argument | Type | Description | Default |
| --- | --- | --- | --- |
| blocks | List[UNetBlockParams] | U-Net blocks | [] |
| include_top | bool | Include top | True |
| use_logits | bool | Use logits | True |
| model_name | str | Model name | "UNet" |
| output_kernel_size | int, tuple[int, int] | Output kernel size | 3 |
| output_kernel_stride | int, tuple[int, int] | Output kernel stride | 1 |

### UNetBlockParams

| Argument | Type | Description | Default |
| --- | --- | --- | --- |
| filters | int | Number of filters | 0 |
| depth | int | Layer depth | 1 |
| ddepth | int | Decoder depth | None |
| kernel | int, tuple[int, int] | Kernel size | 3 |
| pool | int, tuple[int, int] | Pool size | 3 |
| strides | int, tuple[int, int] | Stride size | 1 |
| skip | bool | Add skip connection | True |
| seperable | bool | Use seperable convs | False |
| dropout | float | Dropout rate | None |
| norm | Literal["batch", "layer"] | Normalization type | "batch" |
| dilation | int, tuple[int, int] | Dilation factor | None |

---

## <span class="sk-h2-span">Usage</span>

The following is an example of how to create a model either via CLI or within the `heartkit` python package.

!!! Example

    === "JSON"

        ```json
        {
            "name": "unet",
            "params": {
                "blocks": [
                    {"filters": 12, "depth": 2, "ddepth": 1, "kernel": [1, 5], "pool": [1, 3], "strides": [1, 2], "skip": true, "seperable": true},
                    {"filters": 24, "depth": 2, "ddepth": 1, "kernel": [1, 5], "pool": [1, 3], "strides": [1, 2], "skip": true, "seperable": true},
                    {"filters": 32, "depth": 2, "ddepth": 1, "kernel": [1, 5], "pool": [1, 3], "strides": [1, 2], "skip": true, "seperable": true},
                    {"filters": 48, "depth": 2, "ddepth": 1, "kernel": [1, 5], "pool": [1, 3], "strides": [1, 2], "skip": true, "seperable": true}
                ],
                "output_kernel_size": [1, 5],
                "include_top": true,
                "use_logits": true,
                "model_name": "efficientnetv2"
            }
        }
        ```

    === "Python"

        ```python
        import keras
        from heartkit.models import UNet, UNetParams, UNetBlockParams

        inputs = keras.Input(shape=(800, 1))
        num_classes = 5

        model = UNet(
            x=inputs,
            params=UNetParams(
                blocks=[
                    UNetBlockParams(filters=12, depth=2, ddepth=1, kernel=(1, 5), pool=(1, 3), strides=(1, 2), skip=True, seperable=True),
                    UNetBlockParams(filters=24, depth=2, ddepth=1, kernel=(1, 5), pool=(1, 3), strides=(1, 2), skip=True, seperable=True),
                    UNetBlockParams(filters=32, depth=2, ddepth=1, kernel=(1, 5), pool=(1, 3), strides=(1, 2), skip=True, seperable=True),
                    UNetBlockParams(filters=48, depth=2, ddepth=1, kernel=(1, 5), pool=(1, 3), strides=(1, 2), skip=True, seperable=True)
                ],
                output_kernel_size=(1, 5),
                include_top=True,
                use_logits=True,
                model_name="unet"
            ),
            num_classes=num_classes,
        )
        ```

---
