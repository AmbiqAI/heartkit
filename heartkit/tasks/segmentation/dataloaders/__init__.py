"""
# Segmentation Dataloaders API

Classes:
    Icentia11kDataloader: Icentia 11k dataloader
    LudbDataloader: Ludb dataloader
    PtbxlDataloader: PTB-XL dataloader
    EcgSyntheticDataloader: ECG Synthetic dataloader
    PPgSyntheticDataloader: PPG Synthetic dataloader

"""

import helia_edge as helia

from ....datasets import HKDataloader

from .icentia11k import Icentia11kDataloader
from .ludb import LudbDataloader
from .ptbxl import PtbxlDataloader
from .ecg_synthetic import EcgSyntheticDataloader
from .ppg_synthetic import PPgSyntheticDataloader

SegmentationDataloaderFactory = helia.utils.create_factory(factory="HKSegmentationDataloaderFactory", type=HKDataloader)
SegmentationDataloaderFactory.register("icentia11k", Icentia11kDataloader)
SegmentationDataloaderFactory.register("ludb", LudbDataloader)
SegmentationDataloaderFactory.register("ptbxl", PtbxlDataloader)
SegmentationDataloaderFactory.register("ecg-synthetic", EcgSyntheticDataloader)
SegmentationDataloaderFactory.register("ppg-synthetic", PPgSyntheticDataloader)

__all__ = []
