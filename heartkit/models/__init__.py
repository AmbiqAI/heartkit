"""ModelFactory is used to store and retrieve model generators.
key (str): Model name slug (e.g. "unet")
value (ModelFactoryItem): Model generator
"""

from typing import Protocol

import keras
import neuralspot_edge as nse


class ModelFactoryItem(Protocol):
    """ModelFactoryItem is a protocol for model factory items.

    Args:
        x (keras.KerasTensor): Input tensor
        params (dict): Model parameters
        num_classes (int): Number of classes

    Returns:
        keras.Model: Model
    """

    def __call__(self, x: keras.KerasTensor, params: dict, num_classes: int) -> keras.Model: ...


ModelFactory = nse.utils.ItemFactory[ModelFactoryItem].shared("HKModelFactory")

ModelFactory.register("unet", nse.models.unet.unet_from_object)
ModelFactory.register("unext", nse.models.unext.unext_from_object)
ModelFactory.register("resnet", nse.models.resnet.resnet_from_object)
ModelFactory.register("efficientnetv2", nse.models.efficientnet.efficientnetv2_from_object)
ModelFactory.register("mobileone", nse.models.mobileone.mobileone_from_object)
ModelFactory.register("tcn", nse.models.tcn.tcn_from_object)
ModelFactory.register("composer", nse.models.composer.composer_from_object)
