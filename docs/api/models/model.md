# Model API

HeartKit leverages [neuralspot-edge](https://ambiqai.github.io/neuralspot-edge/) for customizable model architectures. Currently, the models are built using Keras functional model API to allow the most flexibilty in creating custom network topologies. Instead of registering custom `keras.Model` objects, the factory provides a callable that takes a `keras.Input`, model parameters, and number of classes as arguments and returns a `keras.Model`.

## hk.models.ModelFactoryItem

::: heartkit.models.ModelFactoryItem

See [Models](../../models/index.md) for information about available models.
