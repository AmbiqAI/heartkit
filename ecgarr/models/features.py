from typing import Optional
import tensorflow as tf
from .resnet1d import ResNet, BottleneckBlock
from ..types import ArchitectureType

def ecg_feature_extractor(arch: Optional[ArchitectureType] = None, stages: Optional[int] = None):
    """AI based feature extractor. Currently consists of 1D variant of ResNet

    Args:
        arch (str, optional): Architecture name. Defaults to None.
        stages (int, optional): Number of stages. Defaults to None.

    Returns:
        tf.keras.Sequential: Feature extractor model
    """
    if arch is None or arch == 'resnet12':
        resnet = ResNet(
            num_outputs=None,
            input_conv=(32, 7, 2),
            blocks=(1, 1, 1)[:stages],
            filters=(32, 64, 128),
            kernel_size=(7, 5, 3),
            include_top=False
        )
    elif arch == 'resnet18':
        resnet = ResNet(
            num_outputs=None,
            blocks=(2, 2, 2, 2)[:stages],
            kernel_size=(7, 5, 5, 3),
            include_top=False
        )
    elif arch == 'resnet34':
        resnet = ResNet(
            num_outputs=None,
            blocks=(3, 4, 6, 3)[:stages],
            kernel_size=(7, 5, 5, 3),
            include_top=False
        )
    elif arch == 'resnet50':
        resnet = ResNet(
            num_outputs=None,
            blocks=(3, 4, 6, 3)[:stages],
            kernel_size=(7, 5, 5, 3),
            block_fn=BottleneckBlock,
            include_top=False
        )
    else:
        raise ValueError('unknown architecture: {}'.format(arch))

    feature_extractor = tf.keras.Sequential([
        resnet, tf.keras.layers.GlobalAveragePooling1D()
    ])
    return feature_extractor
