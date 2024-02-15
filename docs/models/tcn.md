# Temporal Convolutional Network (TCN)

## <span class="sk-h2-span">Overview</span>

Temporal convolutional network (TCN) is a type of convolutional neural network (CNN) that is commonly used for sequence modeling tasks such as speech recognition, text generation, and video classification. TCN is a fully convolutional network that consists of a series of dilated causal convolutional layers. The dilated convolutional layers allow TCN to have a large receptive field while maintaining a small number of parameters. TCN is also fully parallelizable, which allows for faster training and inference times.

For more info, refer to the original paper [Temporal Convolutional Networks: A Unified Approach to Action Segmentation](https://doi.org/10.48550/arXiv.1608.08242).

## <span class="sk-h2-span">Additions</span>

The TCN architecture has been modified to allow the following:

* Convolutional pairs can be factorized into depthwise separable convolutions.
* Squeeze and excitation (SE) blocks can be added between convolutional pairs.
* Normalization can be set between batch normalization and layer normalization.
* ReLU is replaced with the approximated ReLU6.

---

## <span class="sk-h2-span">Arguments</span>

The following arguments can be passed to the `TCN` class:

`TcnParams`:

| Argument | Type | Description | Default |
| --- | --- | --- | --- |
| input_kernel | int, tuple[int, int] | Input kernel size | None |
| input_norm | Literal["batch", "layer"] | Input normalization type | "layer" |
| block_type | Literal["lg", "mb", "sm"] | Block type | "mb" |
| blocks | List[TcnBlockParams] | TCN blocks | [] |
| output_kernel | int, tuple[int, int] | Output kernel size | 3 |
| include_top | bool | Include top | True |
| use_logits | bool | Use logits | True |
| model_name | str | Model name | "UNext" |

`TcnBlockParams`:

| Argument | Type | Description | Default |
| --- | --- | --- | --- |
| depth | int | Layer depth | 1 |
| branch | int | Number of branches | 1 |
| filters | int | Number of filters | 0 |
| kernel | int, tuple[int, int] | Kernel size | 3 |
| dilation | int, tuple[int, int] | Dilation rate | 1 |
| ex_ratio | float | Expansion ratio | 1 |
| se_ratio | float | Squeeze and excite ratio | 0 |
| dropout | float | Dropout rate | None |
| norm | Literal["batch", "layer"] | Normalization type | "layer" |
