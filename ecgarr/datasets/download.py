import os

import pydantic_argparse

from ..types import EcgDownloadParams
from .icentia11k import IcentiaDataset
from .ludb import LudbDataset


def download_datasets(params: EcgDownloadParams):
    """Download all specified datasets.

    Args:
        params (EcgDownloadParams): Download parameters
    """
    os.makedirs(params.ds_root_path, exist_ok=True)
    #### Icentia11k Dataset
    if "icentia11k" in params.datasets:
        IcentiaDataset(str(params.ds_root_path)).download(
            num_workers=params.data_parallelism,
            force=params.force,
        )

    if "ludb" in params.datasets:
        LudbDataset(str(params.ds_root_path)).download(
            num_workers=params.data_parallelism,
            force=params.force,
        )


def create_parser():
    """Create CLI parser"""
    return pydantic_argparse.ArgumentParser(
        model=EcgDownloadParams,
        prog="ECG Dataset",
        description="ECG dataset",
    )


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_typed_args()
    download_datasets(args)
