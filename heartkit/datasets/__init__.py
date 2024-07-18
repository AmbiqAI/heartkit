from .augmentation import augment_pipeline
from .bidmc import BidmcDataset
from .dataset import HKDataset
from .defines import PatientGenerator, Preprocessor
from .download import download_datasets
from .icentia11k import IcentiaDataset
from .lsad import LsadDataset
from .ludb import LudbDataset
from .nstdb import NstdbNoise
from .preprocessing import preprocess_pipeline
from .ptbxl import PtbxlDataset
from .qtdb import QtdbDataset
from .synthetic import SyntheticDataset
from .syntheticppg import SyntheticPpgDataset
from .utils import (
    create_dataset_from_data,
    create_interleaved_dataset_from_generator,
    random_id_generator,
    uniform_id_generator,
)
from ..utils import create_factory

DatasetFactory = create_factory(factory="HKDatasetFactory", type=HKDataset)

DatasetFactory.register("bidmc", BidmcDataset)
DatasetFactory.register("synthetic", SyntheticDataset)
DatasetFactory.register("syntheticppg", SyntheticPpgDataset)
DatasetFactory.register("icentia11k", IcentiaDataset)
DatasetFactory.register("lsad", LsadDataset)
DatasetFactory.register("ludb", LudbDataset)
DatasetFactory.register("qtdb", QtdbDataset)
DatasetFactory.register("ptbxl", PtbxlDataset)


__all__ = [
    "download_datasets",
    "HKDataset",
    "IcentiaDataset",
    "LudbDataset",
    "PtbxlDataset",
    "QtdbDataset",
    "SyntheticDataset",
]
