from typing import Callable

import keras

_models: dict[str, Callable[[keras.KerasTensor, dict, int], keras.models.Model]] = {}


class ModelFactory:
    """Model factory enables registering, creating, and listing models. It is a singleton class."""

    @staticmethod
    def register(name: str, model: Callable[[keras.KerasTensor, dict, int], keras.models.Model]) -> None:
        """Register a model

        Args:
            name (str): model name
            model (Callable[[keras.KerasTensor, dict, int], keras.models.Model]): model
        """
        _models[name] = model

    @staticmethod
    def unregister(name: str) -> None:
        """Unregister a model

        Args:
            name (str): model name
        """
        _models.pop(name, None)

    @staticmethod
    def create(
        name: str,
        params: dict,
        inputs: keras.KerasTensor,
        num_classes: int | None = None,
    ) -> keras.models.Model:
        """Create a model

        Args:
            name (str): model name
            params (dict): model parameters
            inputs (keras.KerasTensor): input tensor
            num_classes (int): number of classes

        Returns:
            keras.models.Model: model
        """
        return _models[name](inputs, params, num_classes)

    @staticmethod
    def list() -> list[str]:
        """List registered models

        Returns:
            list[str]: model names
        """
        return list(_models.keys())

    @staticmethod
    def get(name: str) -> Callable[[keras.KerasTensor, dict, int], keras.models.Model]:
        """Get a model

        Args:
            name (str): model name

        Returns:
            Callable[[keras.KerasTensor, dict, int], keras.models.Model]: model
        """
        return _models[name]

    @staticmethod
    def has(name: str) -> bool:
        """Check if a model is registered

        Args:
            name (str): model name

        Returns:
            bool: True if the model is registered
        """
        return name in _models
