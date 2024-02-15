# MobileOne

## <span class="sk-h2-span">Overview</span>

MobileOne is a fully convolutional neural network designed to have minimal latency when running in mobile/edge devices. The architecture consists of a series of depthwise separable convolutions and squeeze and excitation (SE) blocks. The network also uses standard batch normalization and ReLU activations that can be easily fused into the convolutional layers. Lastly, the network uses over-parameterized branches to improve training, yet can be merged into a single branch during inference.

For more info, refer to the original paper [MobileOne: An Improved One millisecond Mobile Backbone](https://doi.org/10.48550/arXiv.2206.04040).

---

## <span class="sk-h2-span">Additions</span>

The MobileOne architecture has been modified to allow the following:

* Enable 1D and 2D variants.
* Enable dilated convolutions.

---

## <span class="sk-h2-span">Arguments</span>

The following arguments can be passed to the `MobileOne` class:

`MobileOneParams`:

| Argument | Type | Description | Default |
| --- | --- | --- | --- |
| blocks | List[MobileOneBlockParams] | MobileOne blocks | [] |
| input_filters | int | Input filters | 3 |
| input_kernel_size | int, tuple[int, int] | Input kernel size | 3 |
| input_strides | int, tuple[int, int] | Input stride | 2 |
| input_padding | int, tuple[int, int] | Input padding | 1 |
| include_top | bool | Include top | True |
| dropout | float | Dropout rate | 0.2 |
| model_name | str | Model name | "MobileOne" |

`MobileOneBlockParams`:

| Argument | Type | Description | Default |
| --- | --- | --- | --- |
| filters | int | Number of filters | 0 |
| depth | int | Layer depth | 1 |
| kernel_size | int, tuple[int, int] | Kernel size | 3 |
| strides | int, tuple[int, int] | Stride size | 1 |
| padding | int, tuple[int, int] | Padding size | 0 |
| se_ratio | float | Squeeze Excite ratio | 8 |
| se_depth | int | Depth length to apply SE | 0 |
| num_conv_branches | int | Number of conv branches | 2 |
