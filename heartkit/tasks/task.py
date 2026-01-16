import abc
import os

import helia_edge as helia

from ..datasets import DatasetFactory, HKDataset
from ..defines import HKTaskParams


class HKTask(abc.ABC):
    """heartKIT Task base class. All tasks should inherit from this class."""

    @staticmethod
    def description() -> str:
        """Get task description

        Returns:
            str: description

        """
        return ""

    @staticmethod
    def download(params: HKTaskParams) -> None:
        """Download datasets

        Args:
            params (HKTaskParams): Task parameters

        """
        os.makedirs(params.job_dir, exist_ok=True)
        logger = helia.utils.setup_logger(__name__, level=params.verbose, file_path=params.job_dir / "download.log")
        logger.debug(f"Creating working directory in {params.job_dir}")

        for ds in params.datasets:
            if DatasetFactory.has(ds.name):
                logger.debug(f"Downloading dataset: {ds.name}")
                Dataset = DatasetFactory.get(ds.name)
                ds: HKDataset = Dataset(**ds.params)
                ds.download(
                    num_workers=params.data_parallelism,
                    force=params.force_download,
                )
            # END IF
        # END FOR

    @staticmethod
    def train(params: HKTaskParams) -> None:
        """Train a model

        Args:
            params (HKTaskParams): Task parameters

        """
        raise NotImplementedError

    @staticmethod
    def evaluate(params: HKTaskParams) -> None:
        """Evaluate a model

        Args:
            params (HKTaskParams): Task parameters

        """
        raise NotImplementedError

    @staticmethod
    def export(params: HKTaskParams) -> None:
        """Export a model

        Args:
            params (HKTaskParams): Task parameters

        """
        raise NotImplementedError

    @staticmethod
    def demo(params: HKTaskParams) -> None:
        """Run a demo

        Args:
            params (HKTaskParams): Task parameters

        """
        raise NotImplementedError
