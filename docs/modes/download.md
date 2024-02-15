# Download Datasets

The `download` command is used to download all datasets specified. Please refer to [Datasets](../datasets/index.md) for details on the available datasets. Additional datasets can be added by creating a new dataset class and registering it with __HeartKit__ dataset factory.

## <span class="sk-h2-span">Usage</span>

!!! Example

    The following command will download and prepare four datasets.

    === "CLI"

        ```bash
        heartkit -m download -c ./configs/download-datasets.json
        # ^ No task is required
        ```

    === "Python"

        ```python
        from pathlib import Path
        import heartkit as hk

        hk.datasets.download_datasets(hk.HKDownloadParams(
            ds_path=Path("./datasets"),
            datasets=["icentia11k", "ludb", "qtdb", "synthetic"],
            progress=True
        ))
        ```


## <span class="sk-h2-span">Arguments </span>

The following table lists the arguments that can be used with the `download` command. All datasets will be saved in their own subdirectory within the `ds_path` directory.


| Argument | Type | Opt/Req | Default | Description |
| --- | --- | --- | --- | --- |
| ds_path | Path | Optional | `Path()` | Dataset root directory |
| datasets | list[DatasetTypes] | Optional |  | Datasets |
| progress | bool | Optional | True | Display progress bar |
| force | bool | Optional | False | Force download dataset- overriding existing files |
| data_parallelism | int | Optional | `lambda: os.cpu_count() or 1` | # of data loaders running in parallel |
