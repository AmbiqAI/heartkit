from ..datasets import DatasetFactory, HKDataset
from ..defines import DatasetParams


def load_datasets(
    datasets: list[DatasetParams] = None,
) -> list[HKDataset]:
    """Load datasets

    Args:
        datasets (list[DatasetParams]): List of datasets

    Returns:
        HKDataset: Dataset
    """
    dsets = []
    for dset in datasets:
        if DatasetFactory.has(dset.name):
            dsets.append(DatasetFactory.get(dset.name)(ds_path=dset.path, **dset.params))
        # END IF
    # END FOR
    return dsets
