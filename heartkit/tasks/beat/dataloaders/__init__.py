"""
# Task Dataloaders API

Classes:
    BeatTaskFactory: Beat task factory
    Icentia11kDataloader: Icentia 11k dataloader
    IcentiaMiniDataloader: Icentia Mini dataloader

"""

import helia_edge as helia

from ....datasets import HKDataloader

from .icentia11k import Icentia11kDataloader
from .icentia_mini import IcentiaMiniDataloader

BeatTaskFactory = helia.utils.create_factory(factory="HKBeatTaskFactory", type=HKDataloader)
BeatTaskFactory.register("icentia11k", Icentia11kDataloader)
BeatTaskFactory.register("icentia_mini", IcentiaMiniDataloader)
