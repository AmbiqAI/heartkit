"""DatasetFactory is used to store and retrieve datasets that inherit from HKDataset.
key (str): Dataset name slug (e.g. "ptbxl")
value (HKDataset): Dataset class
"""

import neuralspot_edge as nse

from .dataset import HKDataset

DatasetFactory = nse.utils.create_factory(factory="HKDatasetFactory", type=HKDataset)
