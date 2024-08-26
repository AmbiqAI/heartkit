# Download Datasets

## <span class="sk-h2-span">Introduction</span>

The `download` command is used to download specified datasets. Please refer to [Datasets](../datasets/index.md) for details on the available datasets. Additional datasets can be added by creating a new dataset class and registering it with __HeartKit__ dataset factory, [DatasetFactory](../datasets/index.md#dataset-factory).

## <span class="sk-h2-span">Usage</span>

### CLI

Using the CLI, the `download` command can be used to download specified datasets in the configuration file or directly in the command line.

```bash
heartkit -m download -c '{"datasets": [{"name": "ptbxl", "parameters": {"path": ".datatasets/ptbxl"}}]}'
```

### Python

In code, the `download` method of a dataset can be used to download the dataset.

```py linenums="1"
import heartkit as hk

ds = hk.DatasetFactory.get("ptbxl")(path="./datasets/ptbxl")
ds.download()

```

Similarly, to download multiple datasets, the `download` method of a task can be used.

```py linenums="1"
import heartkit as hk

task = hk.TaskFactory.get("rhythm")

params = hk.HKTaskParams(
    datasets=[hk.NamedParams(
        name="ptbxl",
        parameters={"path": "./datasets/ptbxl"}
    ), hk.NamedParams(
        name="lsad",
        parameters={"path": "./datasets/lsad"}
    )],
    force_download=False
)

task.download(params)

```

## <span class="sk-h2-span">Arguments </span>

Please refer to [HKTaskParams](../modes/configuration.md#hktaskparams) for the available configuration options for the `download` command.
