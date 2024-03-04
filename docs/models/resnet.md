# ResNet

### Overview

ResNet is a type of convolutional neural network (CNN) that is commonly used for image classification tasks. ResNet is a fully convolutional network that consists of a series of convolutional layers and pooling layers. The pooling layers are used to downsample the input while the convolutional layers are used to upsample the input. The skip connections between the pooling layers and convolutional layers allow ResNet to preserve spatial/temporal information while also allowing for faster training and inference times.

For more info, refer to the original paper [Deep Residual Learning for Image Recognition](https://doi.org/10.1109/CVPR.2016.90).

---

### Additions

* Enable 1D and 2D variants.

---

## <span class="sk-h2-span">Arguments</span>

The following arguments can be passed to the `ResNet` class:

### ResnetParams

| Argument | Type | Description | Default |
| --- | --- | --- | --- |
| blocks | List[ResNetBlockParams] | ResNet blocks | [] |
| input_filters | int | Input filters | 0 |
| input_kernel_size | int, tuple[int, int] | Input kernel size | 3 |
| input_strides | int, tuple[int, int] | Input stride | 2 |
| include_top | bool | Include top | True |
| dropout | float | Dropout rate | 0.2 |
| model_name | str | Model name | "ResNet" |

### ResNetBlockParams

| Argument | Type | Description | Default |
| --- | --- | --- | --- |
| filters | int | Number of filters | 0 |
| depth | int | Layer depth | 1 |
| kernel_size | int, tuple[int, int] | Kernel size | 3 |
| strides | int, tuple[int, int] | Stride size | 1 |
| bottleneck | bool | Use bottleneck blocks | False |

---
