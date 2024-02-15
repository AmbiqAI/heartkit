import os

from ..defines import HKDownloadParams
from .factory import DatasetFactory


def download_datasets(params: HKDownloadParams):
    """Download all specified datasets.

    Args:
        params (HeartDownloadParams): Download parameters
    """
    os.makedirs(params.ds_path, exist_ok=True)

    for dataset in params.datasets:
        if DatasetFactory.has(dataset):
            DatasetFactory.get(dataset).download(
                num_workers=params.data_parallelism,
                force=params.force,
            )
