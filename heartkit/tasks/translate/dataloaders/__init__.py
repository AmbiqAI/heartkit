import neuralspot_edge as nse

from ....datasets import HKDataloader

from .bidmc import BidmcDataloader

TranslateTaskFactory = nse.utils.create_factory(factory="HKTranslateTaskFactory", type=HKDataloader)
TranslateTaskFactory.register("bidmc", BidmcDataloader)
