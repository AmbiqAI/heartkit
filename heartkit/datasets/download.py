import logging
import os

from ..defines import HKDownloadParams
from ..utils import setup_logger
from . import DatasetFactory

logger = setup_logger(__name__)


def download_datasets(params: HKDownloadParams):
    """Download specified datasets.

    Args:
        params (HeartDownloadParams): Download parameters

    """
    os.makedirs(params.job_dir, exist_ok=True)
    logger.debug(f"Creating working directory in {params.job_dir}")

    handler = logging.FileHandler(params.job_dir / "download.log", mode="w")
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)

    for ds in params.datasets:
        if DatasetFactory.has(ds.name):
            os.makedirs(ds.path, exist_ok=True)
            Dataset = DatasetFactory.get(ds.name)
            ds = Dataset(ds_path=ds.path, **ds.params)
            ds.download(
                num_workers=params.data_parallelism,
                force=params.force,
            )
        # END IF
    # END FOR
