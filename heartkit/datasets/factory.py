from typing import Type

from .dataset import HKDataset

_datasets: dict[str, Type[HKDataset]] = {}


class DatasetFactory:
    """Dataset factory enables registering, creating, and listing datasets. It is a singleton class."""

    @staticmethod
    def register(name: str, dataset: Type[HKDataset]) -> None:
        """Register a dataset

        Args:
            name (str): dataset name
            dataset (HKDataset): dataset
        """
        _datasets[name] = dataset

    @staticmethod
    def create(name: str, **kwargs) -> HKDataset:
        """Create a dataset

        Args:
            name (str): dataset name

        Returns:
            HKDataset: dataset
        """
        return _datasets[name](**kwargs)

    @staticmethod
    def list() -> list[str]:
        """List registered datasets

        Returns:
            list[str]: dataset names
        """
        return list(_datasets.keys())

    @staticmethod
    def get(name: str) -> Type[HKDataset]:
        """Get a dataset

        Args:
            name (str): dataset name

        Returns:
            HKDataset: dataset
        """
        return _datasets[name]

    @staticmethod
    def has(name: str) -> bool:
        """Check if a dataset is registered

        Args:
            name (str): dataset name

        Returns:
            bool: True if dataset is registered
        """
        return name in _datasets
