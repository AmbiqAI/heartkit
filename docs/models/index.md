# :factory: Model Factory

HeartKit provides a model factory that allows you to easily create and train customized models via [KerasEdge](). KerasEdge includes a growing number of state-of-the-art models that can be easily configured and trained using high-level parameters. The models are designed to be efficient and well-suited for real-time edge applications. Most of the models are based on state-of-the-art architectures that have been modified to allow for more fine-grain customization. The also support 1D variants to allow for training on time-series data. The included models are well suited for efficient, real-time edge applications.

Please check [KerasEdge]() for list of available models and their configurations.

---

## <span class="sk-h2-span">Usage</span>

The model factory can be invoked either via CLI or within the `heartkit` python package. At a high level, the model factory performs the following actions based on the provided configuration parameters:

!!! Example

    === "JSON"

        ```json
        {
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
        ```

    === "Python"

        ```python
        import keras
        from heartkit.models import Tcn, TcnParams, TcnBlockParams

        inputs = keras.Input(shape=(800, 1))
        num_classes = 5

        model = Tcn(
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

---
