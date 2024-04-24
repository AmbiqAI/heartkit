# Bring-Your-Own-Dataset (BYOD)

The Bring-Your-Own-Dataset (BYOD) feature allows users to add custom datasets for training and evaluating models. This feature is useful when working with proprietary or custom datasets that are not available in the HeartKit library.

## How it Works

1. **Create a Dataset**: Define a new dataset by creating a new Python file. The file should contain a class that inherits from the `HKDataset` base class and implements the required methods.

    ```python
    import heartkit as hk

    class CustomDataset(hk.HKDataset):
        def __init__(self, config):
            super().__init__(config)

        def download(self):
            pass

        def generate(self):
            pass
    ```

2. **Register the Dataset**: Register the new dataset with the `DatasetFactory` by calling the `register` method. This method takes the dataset name and the dataset class as arguments.

    ```python
    import heartkit as hk

    hk.DatasetFactory.register("custom", CustomDataset)
    ```

3. **Use the Dataset**: The new dataset can now be used with the `DatasetFactory` to perform various operations such as downloading and generating data.

    ```python
    import heartkit as hk

    dataset = hk.DatasetFactory.create("custom", config)
    ```
