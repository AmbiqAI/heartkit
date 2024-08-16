from .augmentation import create_augmentation_pipeline
from .bidmc import BidmcDataset
from .dataset import HKDataset
from .defines import PatientGenerator
from .dataloader import HKDataloader
from .icentia11k import IcentiaDataset, IcentiaBeat, IcentiaRhythm
from .icentia_mini import IcentiaMiniDataset, IcentiaMiniRhythm, IcentiaMiniBeat
from .lsad import LsadDataset, LsadScpCode
from .ludb import LudbDataset, LudbSegmentation
from .nstdb import NstdbNoise
from .ptbxl import PtbxlDataset, PtbxlScpCode
from .qtdb import QtdbDataset
from .ecg_synthetic import EcgSyntheticDataset
from .ppg_synthetic import PpgSyntheticDataset
from .factory import DatasetFactory

DatasetFactory.register("bidmc", BidmcDataset)
DatasetFactory.register("ecg-synthetic", EcgSyntheticDataset)
DatasetFactory.register("ppg-synthetic", PpgSyntheticDataset)
DatasetFactory.register("icentia11k", IcentiaDataset)
DatasetFactory.register("icentia_mini", IcentiaMiniDataset)
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
    "EcgSyntheticDataset",
    "NstdbNoise",
]
