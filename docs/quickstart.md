# Quickstart
<!-- # :octicons-heart-fill-24:{ .heart } Overview -->

## <span class="sk-h2-span">Install HeartKit</span>

We provide several installation methods including pip, poetry, and Docker. Install __HeartKit__ via pip/poetry for the latest stable release or by cloning the GitHub repo for the most up-to-date. Additionally, a [VSCode Dev Container](https://code.visualstudio.com/docs/devcontainers/containers) is available and defined in [./.devcontainer](https://github.com/AmbiqAI/heartkit/tree/main/.devcontainer) to run in an isolated Docker environment.

!!! install

    === "Git clone"

        Clone the repository if you are interested in contributing to the development or wish to experiment with the latest source code. After cloning, navigate into the directory and install the package. In this mode, Poetry is recommended.

        ```bash
        # Clone the repository
        git clone https://github.com/AmbiqAI/heartkit.git

        # Navigate to the cloned directory
        cd heartkit

        # Install the package in editable mode for development
        poetry install
        ```

        When using editable mode via Poetry, be sure to activate the python environment: `poetry shell`. <br>
        On Windows using Powershell, use `.venv\Scripts\activate.ps1`.

    === "PyPI install"

        Install the HeartKit package using pip or Poetry.
        Visit the Python Package Index (PyPI) for more details on the package: [https://pypi.org/project/heartkit/](https://pypi.org/project/heartkit/)

        ```bash
        # Install with pip
        pip install heartkit
        ```

        Or, if you prefer to use Poetry, you can install the package with the following command:

        ```bash
        # Install with poetry
        poetry add heartkit
        ```

        Alternatively, you can install the latest development version directly from the GitHub repository. Make sure to have the Git command-line tool installed on your system. The @main command installs the main branch and may be modified to another branch, i.e. @canary.

        ```bash
        pip install git+https://github.com/AmbiqAI/heartkit.git@main
        ```

        Or, using Poetry:

        ```bash
        poetry add git+https://github.com/AmbiqAI/heartkit.git@main
        ```


## <span class="sk-h2-span">Requirements</span>

* [Python ^3.11+](https://www.python.org)
* [Poetry ^1.6.1+](https://python-poetry.org/docs/#installation)

Check the project's [pyproject.toml](https://github.com/AmbiqAI/heartkit/blob/main/pyproject.toml) file for a list of up-to-date Python dependencies. Note that the installation methods above install all required dependencies. The following are also required to compile and flash the binary to evaluate the demos running on Ambiq's evaluation boards (EVBs):

* [Arm GNU Toolchain ^12.2](https://developer.arm.com/downloads/-/arm-gnu-toolchain-downloads)
* [Segger J-Link ^7.92](https://www.segger.com/downloads/jlink/)

Once installed, __HeartKit__ can be used as either a CLI-based tool or as a Python package to perform advanced experimentation.

---

## <span class="sk-h2-span">Use HeartKit with CLI</span>

The HeartKit command line interface (CLI) allows for simple single-line commands without the need for a Python environment. The CLI requires no customization or Python code. You can simply run all the built-in tasks from the terminal with the __heartkit__ command. Check out the [CLI Guide](./usage/cli.md) to learn more about available options.

!!! example

    === "Syntax"
        Heartkit commands use the following syntax:

        ```bash
        heartkit --mode [MODE] --task [TASK] --config [CONFIG]
        ```

        Or using short flags:

        ```bash
        heartkit -m [MODE] -t [TASK] -c [CONFIG]
        ```

        Where:

        * `MODE` is one of `download`, `train`, `evaluate`, `export`, or `demo`
        * `TASK` is one of `segmentation`, `rhythm`, `beat`, or `denoise`
        * `CONFIG` is configuration as JSON content or file path

    === "Download"
        Download datasets specified in the configuration file.

        ```bash
        heartkit -m download -c ./download-datasets.json
        ```

    === "Train"
        Train a rhythm model using the supplied configuration file.

        ```bash
        heartkit -m train -t rhythm -c ./configuration.json
        ```

    === "Evaluate"
        Evaluate the trained rhythm model using the supplied configuration file.

        ```bash
        heartkit -m evaluate -t rhythm  -c ./configuration.json
        ```

    === "Demo"
        Run demo on trained rhythm model using the supplied configuration file.

        ```bash
        heartkit -m demo -t rhythm -c ./configuration.json
        ```

## <span class="sk-h2-span">Use HeartKit with Python</span>

The __HeartKit__ Python package allows for more fine-grained control and customization. You can use the package to train, evaluate, and deploy models for a variety of tasks. You can create custom datasets, models, and tasks and register them with corresponding factories and use them like built-in tasks.

For example, you can create a custom task, train it, evaluate its performance on a validation set, and even export a quantized TensorFlow Lite model for deployment. Check out the [Python Guide](./usage/python.md) to learn more about using HeartKit as a Python package.

!!! Example

    ```python
    import heartkit as hk

    ds_params = hk.HKDownloadParams(
        ds_path="./datasets",
        datasets=["ludb", "ecg-synthetic"],
        progress=True
    )

    # Download datasets
    hk.datasets.download_datasets(ds_params)

    # Generate task parameters from configuration
    params = hk.HKTaskParams(...)  # Expand to see example (1)

    task = hk.TaskFactory.get("rhythm")

    # Train rhythm model
    task.train(params)

    # Evaluate rhythm model
    task.evaluate(params)

    # Export rhythm model
    task.export(params)

    ```

    1. Configuration parameters:
    --8<-- "assets/usage/python-configuration.md"


---
