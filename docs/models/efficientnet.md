# EfficientNetV2

## <span class="sk-h2-span">Overview</span>

EfficientNetV2 is an improvement to EfficientNet that incorporates additional optimizations to reduce both computation and memory. In particular, the architecture leverages both fused and non-fused MBConv blocks, non-uniform layer scaling, and training-aware NAS.

For more info, refer to the original paper [EfficientNetV2: Smaller Models and Faster Training](https://arxiv.org/abs/2104.00298).

---

## <span class="sk-h2-span">Additions</span>

The EfficientNetV2 architecture has been modified to allow the following:

* Enable 1D and 2D variants.

---

## <span class="sk-h2-span">Arguments</span>

The following arguments can be passed to the `EfficientNetV2` class:

### EfficientNetV2Params

| Argument | Type | Description | Default |
| --- | --- | --- | --- |
| blocks | List[MBConvParams] | EfficientNet blocks | [] |
| input_filters | int | Input filters | 0 |
| input_kernel_size | int, tuple[int, int] | Input kernel size | 3 |
| input_strides | int, tuple[int, int] | Input stride | 2 |
| output_filters | int | Output filters | 0 |
| include_top | bool | Include top | True |
| dropout | float | Dropout rate | 0.2 |
| drop_connect_rate | float | Drop connect rate | 0.2 |
| model_name | str | Model name | "EfficientNetV2" |


### MBConvParams

| Argument | Type | Description | Default |
| --- | --- | --- | --- |
| filters | int | Number of filters | 0 |
| depth | int | Layer depth | 1 |
| ex_ratio | float | Expansion ratio | 1 |
| kernel_size | int, tuple[int, int] | Kernel size | 3 |
| strides | int, tuple[int, int] | Stride size | 1 |
| se_ratio | float | Squeeze Excite ratio | 8 |
| droprate | float | Drop rate | 0 |

## <span class="sk-h2-span">Usage</span>

The following is an example of how to create a model either via CLI or within the `heartkit` python package.

!!! Example

    === "JSON"

        ```json
        {
            "name": "efficientnetv2",
            "params": {

                "input_filters": 24,
                "input_kernel_size": [1, 7],
                "input_stride": [1, 2],
                "blocks": [
                    {"filters": 32, "depth": 2, "kernel_size": [1, 7], "stride": [1, 2], "ex_ratio": 1,  "se_ratio": 2},
                    {"filters": 48, "depth": 2, "kernel_size": [1, 7], "stride": [1, 2], "ex_ratio": 1,  "se_ratio": 2},
                    {"filters": 64, "depth": 2, "kernel_size": [1, 7], "stride": [1, 2], "ex_ratio": 1,  "se_ratio": 2},
                    {"filters": 72, "depth": 1, "kernel_size": [1, 7], "stride": [1, 2], "ex_ratio": 1,  "se_ratio": 2}
                ],
                "output_filters": 0,
                "include_top": true,
                "use_logits": true,
                "model_name": "efficientnetv2"
            }
        }
        ```

    === "Python"

        ```python
        import keras
        from heartkit.models import EfficientNetV2, EfficientNetV2Params, MBConvParams

        inputs = keras.Input(shape=(800, 1))
        num_classes = 5

        model = EfficientNetV2(
            x=inputs,
            params=EfficientNetV2Params(
                input_filters=24,
                input_kernel_size=(1, 7),
                input_strides=(1, 2),
                blocks=[
                    MBConvParams(filters=32, depth=2, kernel_size=(1, 7), strides=(1, 2), ex_ratio=1, se_ratio=2),
                    MBConvParams(filters=48, depth=2, kernel_size=(1, 7), strides=(1, 2), ex_ratio=1, se_ratio=2),
                    MBConvParams(filters=64, depth=2, kernel_size=(1, 7), strides=(1, 2), ex_ratio=1, se_ratio=2),
                    MBConvParams(filters=72, depth=1, kernel_size=(1, 7), strides=(1, 2), ex_ratio=1, se_ratio=2)
                ],
                output_filters=0,
                include_top=True,
                dropout=0.2,
                drop_connect_rate=0.2,
                model_name="efficientnetv2"
            )
            num_classes=num_classes,
        )
        ```

---
