"""
# Foundation Dataloaders API

Classes:
    FoundationTaskFactory: Foundation task factory
    LsadDataloader: LSAD dataloader
    PtbxlDataloader: PTB-XL dataloader

"""

import neuralspot_edge as nse

from ....datasets import HKDataloader

from .lsad import LsadDataloader
from .ptbxl import PtbxlDataloader

FoundationTaskFactory = nse.utils.create_factory(factory="FoundationTaskFactory", type=HKDataloader)
FoundationTaskFactory.register("lsad", LsadDataloader)
FoundationTaskFactory.register("ptbxl", PtbxlDataloader)
