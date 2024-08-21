import neuralspot_edge as nse

from ....datasets import HKDataloader

from .icentia11k import Icentia11kDataloader
from .ludb import LudbDataloader
from .ptbxl import PtbxlDataloader
from .ecg_synthetic import EcgSyntheticDataloader
from .ppg_synthetic import PPgSyntheticDataloader

SegmentationDataloaderFactory = nse.utils.create_factory(factory="HKSegmentationDataloaderFactory", type=HKDataloader)
SegmentationDataloaderFactory.register("icentia11k", Icentia11kDataloader)
SegmentationDataloaderFactory.register("ludb", LudbDataloader)
SegmentationDataloaderFactory.register("ptbxl", PtbxlDataloader)
SegmentationDataloaderFactory.register("ecg-synthetic", EcgSyntheticDataloader)
SegmentationDataloaderFactory.register("ppg-synthetic", PPgSyntheticDataloader)
