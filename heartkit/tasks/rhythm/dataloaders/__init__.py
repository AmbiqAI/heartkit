import neuralspot_edge as nse

from ....datasets import HKDataloader

from .icentia11k import Icentia11kDataloader
from .icentia_mini import IcentiaMiniDataloader
from .ptbxl import PtbxlDataloader
from .lsad import LsadDataloader

RhythmDataloaderFactory = nse.utils.create_factory(factory="HKRhythmDataloaderFactory", type=HKDataloader)
RhythmDataloaderFactory.register("icentia11k", Icentia11kDataloader)
RhythmDataloaderFactory.register("icentia_mini", IcentiaMiniDataloader)
RhythmDataloaderFactory.register("ptbxl", PtbxlDataloader)
RhythmDataloaderFactory.register("lsad", LsadDataloader)
