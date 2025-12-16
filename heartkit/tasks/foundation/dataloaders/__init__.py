"""
# Foundation Dataloaders API

Classes:
    FoundationTaskFactory: Foundation task factory
    LsadDataloader: LSAD dataloader
    PtbxlDataloader: PTB-XL dataloader

"""

import helia_edge as helia

from ....datasets import HKDataloader

from .lsad import LsadDataloader
from .ptbxl import PtbxlDataloader

FoundationTaskFactory = helia.utils.create_factory(factory="FoundationTaskFactory", type=HKDataloader)
FoundationTaskFactory.register("lsad", LsadDataloader)
FoundationTaskFactory.register("ptbxl", PtbxlDataloader)
