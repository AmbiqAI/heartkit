"""
# Translate Dataloaders API

Classes:
    TranslateTaskFactory: Translate task factory
    BidmcDataloader: BIDMC dataloader

"""

import helia_edge as helia

from ....datasets import HKDataloader

from .bidmc import BidmcDataloader

TranslateTaskFactory = helia.utils.create_factory(factory="HKTranslateTaskFactory", type=HKDataloader)
TranslateTaskFactory.register("bidmc", BidmcDataloader)
