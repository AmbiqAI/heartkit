import logging
import os
import neuralspot_edge as nse

from ..defines import HKDownloadParams
from . import HKDataset
from .factory import DatasetFactory


logger = nse.utils.setup_logger(__name__)


def download_datasets(params: HKDownloadParams):
    """Download specified datasets.

    Args:
        params (HKDownloadParams): Download parameters

    Example:
    ```python
    import heartkit as hk

    # Download datasets
    params = hk.HKDownloadParams(
        datasets=[
            hk.NamedParams(name="ptbxl", params={
                "path": "./datasets/ptbxl",
            }),
        ],
        data_parallelism=4,
        force=False,
    )
    hk.datasets.download_datasets(params)
    ```
    """
    os.makedirs(params.job_dir, exist_ok=True)
    logger.debug(f"Creating working directory in {params.job_dir}")

    handler = logging.FileHandler(params.job_dir / "download.log", mode="w")
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)

    for ds in params.datasets:
        if DatasetFactory.has(ds.name):
            Dataset = DatasetFactory.get(ds.name)
            ds: HKDataset = Dataset(**ds.params)
            ds.download(
                num_workers=params.data_parallelism,
                force=params.force,
            )
        # END IF
    # END FOR
