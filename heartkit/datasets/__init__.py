from .augmentation import augment_pipeline, preprocess_pipeline
from .dataset import HKDataset
from .download import download_datasets
from .factory import DatasetFactory
from .icentia11k import IcentiaDataset
from .ludb import LudbDataset
from .qtdb import QtdbDataset
from .synthetic import SyntheticDataset

DatasetFactory.register("synthetic", SyntheticDataset)
DatasetFactory.register("icentia11k", IcentiaDataset)
DatasetFactory.register("ludb", LudbDataset)
DatasetFactory.register("qtdb", QtdbDataset)

__all__ = [
    "download_datasets",
    "HKDataset",
    "IcentiaDataset",
    "LudbDataset",
    "QtdbDataset",
    "SyntheticDataset",
]
