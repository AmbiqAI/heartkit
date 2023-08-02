import os

from ..defines import HeartDownloadParams
from .icentia11k import IcentiaDataset
from .ludb import LudbDataset
from .ptbxl import PtbxlDataset
from .qtdb import QtdbDataset


def download_datasets(params: HeartDownloadParams):
    """Download all specified datasets.

    Args:
        params (HeartDownloadParams): Download parameters
    """
    os.makedirs(params.ds_path, exist_ok=True)

    if "icentia11k" in params.datasets:
        IcentiaDataset(str(params.ds_path)).download(
            num_workers=params.data_parallelism,
            force=params.force,
        )

    if "ludb" in params.datasets:
        LudbDataset(str(params.ds_path)).download(
            num_workers=params.data_parallelism,
            force=params.force,
        )

    if "qtdb" in params.datasets:
        QtdbDataset(str(params.ds_path)).download(
            num_workers=params.data_parallelism,
            force=params.force,
        )

    if "ptbxl" in params.datasets:
        PtbxlDataset(str(params.ds_path)).download(
            num_workers=params.data_parallelism,
            force=params.force,
        )
