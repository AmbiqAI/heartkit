import keras_edge as kedge

from .factory import ModelFactory

ModelFactory.register("unet", kedge.models.unet.unet_from_object)
ModelFactory.register("unext", kedge.models.unext.unext_from_object)
ModelFactory.register("resnet", kedge.models.resnet.resnet_from_object)
ModelFactory.register("efficientnetv2", kedge.models.efficientnet.efficientnetv2_from_object)
ModelFactory.register("mobileone", kedge.models.mobileone.mobileone_from_object)
ModelFactory.register("tcn", kedge.models.tcn.tcn_from_object)
ModelFactory.register("composer", kedge.models.composer.composer_from_object)
