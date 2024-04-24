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

        --8<-- "assets/modes/python-download-snippet.md"


## <span class="sk-h2-span">Arguments </span>

The following table lists the arguments that can be used with the `download` command. All datasets will be saved in their own subdirectory within the `ds_path` directory.


--8<-- "assets/modes/download-params.md"
