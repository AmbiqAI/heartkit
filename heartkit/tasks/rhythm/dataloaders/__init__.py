"""
# Rhythm Dataloaders API

Classes:
    RhythmDataloaderFactory: Rhythm dataloader factory
    Icentia11kDataloader: Icentia 11k dataloader
    IcentiaMiniDataloader: Icentia Mini dataloader
    PtbxlDataloader: PTB-XL dataloader
    LsadDataloader: LSAD dataloader

"""

import helia_edge as helia

from ....datasets import HKDataloader

from .icentia11k import Icentia11kDataloader
from .icentia_mini import IcentiaMiniDataloader
from .ptbxl import PtbxlDataloader
from .lsad import LsadDataloader

RhythmDataloaderFactory = helia.utils.create_factory(factory="HKRhythmDataloaderFactory", type=HKDataloader)
RhythmDataloaderFactory.register("icentia11k", Icentia11kDataloader)
RhythmDataloaderFactory.register("icentia_mini", IcentiaMiniDataloader)
RhythmDataloaderFactory.register("ptbxl", PtbxlDataloader)
RhythmDataloaderFactory.register("lsad", LsadDataloader)
