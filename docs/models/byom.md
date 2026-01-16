# Bring-Your-Own-Model (BYOM)

The model factory can be extended to include custom models. This is useful when you have a custom model architecture that you would like to use for training. The custom model can be registered with the model factory by defining a custom model function and registering it with the `ModelFactory`.

## How it Works

1. **Create a Model**: Define a new model function that takes a `keras.Input`, model parameters, and number of classes as arguments and returns a `keras.Model`.

    ```py linenums="1"

    import keras
    import heartkit as hk

    def custom_model_from_object(
        x: keras.KerasTensor,
        params: dict,
        num_classes: int | None = None,
    ) -> keras.Model:

        y = x
        # Create fully connected network from params
        for layer in params["layers"]:
            y = keras.layers.Dense(layer["units"], activation=layer["activation"])(y)

        if num_classes:
            y = keras.layers.Dense(num_classes, activation="softmax")(y)

        return keras.Model(inputs=x, outputs=y)
    ```

2. **Register the Model**: Register the new model function with the `ModelFactory` by calling the `register` method. This method takes the model name and the callable as arguments.

    ```py linenums="1"
    hk.ModelFactory.register("custom-model", custom_model_from_object)
    ```

3. **Use the Model**: The new model can now be used with the `ModelFactory` to perform various operations such as downloading and generating data.

    ```py linenums="1"
    inputs = keras.Input(shape=(100,))
    model = hk.ModelFactory.get("custom-model")(
        inputs=inputs,
        params={
            "layers": [
                {"units": 64, "activation": "relu"},
                {"units": 32, "activation": "relu"},
            ]
        },
        num_classes=5,
    )

    model.summary()

    ```

## Better Model Params

Rather than using a dictionary to define the model parameters, you can define a custom dataclass or [Pydantic](https://pydantic-docs.helpmanual.io/) model to enforce type checking and provide better documentation.

```py linenums="1"
from pydantic import BaseModel

class CustomLayerParams(BaseModel):
    units: int
    activation: str

class CustomModelParams(BaseModel):
    layers: list[CustomLayerParams]

def custom_model_from_object(
    x: keras.KerasTensor,
    params: dict,
    num_classes: int | None = None,
) -> keras.Model:

    # Convert and validate params
    params = CustomModelParams(**params)

    y = x
    # Create fully connected network from params
    for layer in params.layers:
        y = keras.layers.Dense(layer.units, activation=layer.activation)(y)

    if num_classes:
        y = keras.layers.Dense(num_classes, activation="softmax")(y)

    return keras.Model(inputs=x, outputs=y)
```
