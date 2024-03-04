## <span class="sk-h2-span">U-NeXt </span>

### Overview

U-NeXt is a modification of U-Net that utilizes techniques from ResNeXt and EfficientNetV2. During the encoding phase, mbconv blocks are used to efficiently process the input.

---

### Additions

The U-NeXt architecture has been modified to allow the following:

* MBConv blocks used in the encoding phase.
* Squeeze and excitation (SE) blocks added within blocks.

---

## <span class="sk-h2-span">Arguments</span>

The following arguments can be passed to the `U-NeXt` class:

`UNextParams`:

| Argument | Type | Description | Default |
| --- | --- | --- | --- |
| blocks | List[UNextBlockParams] | U-NeXt blocks | [] |
| include_top | bool | Include top | True |
| use_logits | bool | Use logits | True |
| model_name | str | Model name | "UNext" |
| output_kernel_size | int, tuple[int, int] | Output kernel size | 3 |
| output_kernel_stride | int, tuple[int, int] | Output kernel stride | 1 |

`UNextBlockParams`:

| Argument | Type | Description | Default |
| --- | --- | --- | --- |
| filters | int | Number of filters | 0 |
| depth | int | Layer depth | 1 |
| ddepth | int | Decoder depth | None |
| kernel | int, tuple[int, int] | Kernel size | 3 |
| pool | int, tuple[int, int] | Pool size | 3 |
| strides | int, tuple[int, int] | Stride size | 1 |
| skip | bool | Add skip connection | True |
| expand_ratio | float | Expansion ratio | 1 |
| se_ratio | float | Squeeze and excite ratio | 0 |
| dropout | float | Dropout rate | None |
| norm | Literal["batch", "layer"] | Normalization type | "layer" |

---

## <span class="sk-h2-span">References</span>

* [U2-Net: Going Deeper with Nested U-Structure for Salient Object Detection](https://arxiv.org/abs/2005.09007)
* [UNET 3+: A FULL-SCALE CONNECTED UNET FOR MEDICAL IMAGE SEGMENTATION](https://arxiv.org/pdf/2004.08790.pdf)
* [ResUNet-a: a deep learning framework for semantic segmentation of remotely sensed data](https://arxiv.org/pdf/1904.00592.pdf)
