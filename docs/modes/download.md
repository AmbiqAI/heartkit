# Download Datasets

## <span class="sk-h2-span">Introduction</span>

The `download` command is used to download all datasets specified. Please refer to [Datasets](../datasets/index.md) for details on the available datasets. Additional datasets can be added by creating a new dataset class and registering it with __HeartKit__ dataset factory.

## <span class="sk-h2-span">Usage</span>

### CLI

Using the CLI, the `download` command can be used to download specified datasets in the configuration file or directly in the command line.

```bash
heartkit -m download -c '{"datasets": [{"name": "ptbxl", "parameters": {"path": ".datatasets/ptbxl"}}]}'
```

### Python

Using HeartKit in Python, the `download` method can be used for a specific dataset.

```python
import heartkit as hk

ds = hk.DatasetFactory.get("ptbxl")(path=".datasets/ptbxl")
ds.download()
```

To download multiple datasets, the high-level `download_datasets` function can be used.

```python
import heartkit as hk

params = hk.HKDownloadParams(
    ds_path="./datasets",
    datasets=[hk.NamedParams(
        name="ptbxl",
        parameters={"path": ".datasets/ptbxl"}
    ), hk.NamedParams(
        name="lsad",
        parameters={"path": ".datasets/lsad"}
    )]
    progress=True
)

hk.datasets.download_datasets(params)
```

## <span class="sk-h2-span">Arguments </span>

Please refer to [HKDownloadParams](../modes/configuration.md#hkdownloadparams) for the list of arguments that can be used with the `download` command.
