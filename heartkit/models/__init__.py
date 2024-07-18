from typing import Protocol

import keras
import neuralspot_edge as nse

from ..utils import ItemFactory


class ModelFactoryItem(Protocol):
    def __call__(self, x: keras.KerasTensor, params: dict, num_classes: int) -> keras.Model: ...


ModelFactory = ItemFactory[ModelFactoryItem].shared("HKModelFactory")

ModelFactory.register("unet", nse.models.unet.unet_from_object)
ModelFactory.register("unext", nse.models.unext.unext_from_object)
ModelFactory.register("resnet", nse.models.resnet.resnet_from_object)
ModelFactory.register("efficientnetv2", nse.models.efficientnet.efficientnetv2_from_object)
ModelFactory.register("mobileone", nse.models.mobileone.mobileone_from_object)
ModelFactory.register("tcn", nse.models.tcn.tcn_from_object)
ModelFactory.register("composer", nse.models.composer.composer_from_object)
