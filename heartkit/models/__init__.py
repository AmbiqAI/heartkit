from .defines import MBConvParams
from .efficientnet import EfficientNetParams, EfficientNetV2, efficientnetv2_from_object
from .factory import ModelFactory
from .mobileone import MobileOne, MobileOneU0, mobileone_from_object
from .multiresnet import MultiresNet, MultiresNetParams, multiresnet_from_object
from .resnet import ResNet, ResNetBlockParams, ResNetParams, resnet_from_object
from .tcn import Tcn, TcnBlockParams, TcnParams, tcn_from_object
from .tsmixer import TsBlockParams, TsMixer, TsMixerParams, tsmixer_from_object
from .unet import UNet, UNetBlockParams, UNetParams, unet_from_object
from .unext import UNext, UNextBlockParams, UNextParams, unext_from_object

ModelFactory.register("unet", unet_from_object)
ModelFactory.register("unext", unext_from_object)
ModelFactory.register("resnet", resnet_from_object)
ModelFactory.register("multiresnet", multiresnet_from_object)
ModelFactory.register("efficientnetv2", efficientnetv2_from_object)
ModelFactory.register("mobileone", mobileone_from_object)
ModelFactory.register("tsmixer", tsmixer_from_object)
ModelFactory.register("tcn", tcn_from_object)
