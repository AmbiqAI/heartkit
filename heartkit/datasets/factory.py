"""DatasetFactory is used to store and retrieve datasets that inherit from HKDataset.
key (str): Dataset name slug (e.g. "ptbxl")
value (HKDataset): Dataset class
"""

import helia_edge as helia

from .dataset import HKDataset

DatasetFactory = helia.utils.create_factory(factory="HKDatasetFactory", type=HKDataset)
