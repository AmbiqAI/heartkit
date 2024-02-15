# Quickstart
<!-- # :octicons-heart-fill-24:{ .heart } Overview -->

## <span class="sk-h2-span">Install HeartKit</span>

We provide several installation methods including pip, poetry, and Docker. Install __HeartKit__ via pip/poetry for the latest stable release or by cloning the GitHub repo for the most up-to-date. Additionally, a [VSCode Dev Container](https://code.visualstudio.com/docs/devcontainers/containers) is available and defined in [./.devcontainer](https://github.com/AmbiqAI/heartkit/tree/main/.devcontainer) to run in an isolated Docker environment.

!!! install

    === "Pip/Poetry install"

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

        Alternatively, you can install the latest development version directly from the GitHub repository. Make sure to have the Git command-line tool installed on your system. The @main command installs the main branch and may be modified to another branch, i.e. @release.

        ```bash
        pip install git+https://github.com/AmbiqAI/heartkit.git@main
        ```

    === "Git clone"

        Clone the repository if you are interested in contributing to the development or wish to experiment with the latest source code. After cloning, navigate into the directory and install the package. In this mode, Poetry is recommended.

        ```bash
        # Clone the ultralytics repository
        git clone https://github.com/AmbiqAI/heartkit.git

        # Navigate to the cloned directory
        cd heartkit

        # Install the package in editable mode for development
        poetry install
        ```

## <span class="sk-h2-span">Requirements</span>

* [Python ^3.11+](https://www.python.org)
* [Poetry ^1.6.1+](https://python-poetry.org/docs/#installation)

Check the project's [pyproject.toml](https://github.com/AmbiqAI/heartkit/blob/main/pyproject.toml) file for a list of up-to-date Python dependencies. Note that the installation methods above install all required dependencies. The following are also required to compile and flash the binary to evaluate the demos running on Ambiq's evaluation boards (EVB):

* [Arm GNU Toolchain ^12.2](https://developer.arm.com/downloads/-/arm-gnu-toolchain-downloads)
* [Segger J-Link ^7.92](https://www.segger.com/downloads/jlink/)

Once installed, __HeartKit__ can be used as either a CLI-based tool or as a Python package to perform advanced experimentation.

---

## <span class="sk-h2-span">Use HeartKit with CLI</span>

The HeartKit command line interface (CLI) allows for simple single-line commands without the need for a Python environment. CLI requires no customization or Python code. You can simply run all tasks from the terminal with the __heartkit__ command. Check out the [CLI Guide](./usage/cli.md) to learn more about available options.

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
        * `TASK` is one of `segmentation`, `arrhythmia`, `beat`, or `denoise`
        * `CONFIG` is configuration as JSON content or file path

    === "Download"
        Download datasets specified in the configuration file.

        ```bash
        heartkit -m download -c ./configs/download-datasets.json
        ```

    === "Train"
        Train a 2-class arrhythmia model using the supplied configuration file.

        ```bash
        heartkit -m train -t arrhythmia -c ./configs/arrhythmia-class-2.json
        ```

    === "Evaluate"
        Evaluate the trained arrhythmia model using the same configuration file.

        ```bash
        heartkit -m evaluate -t arrhythmia  -c ./configs/arrhythmia-class-2.json
        ```

    === "Demo"
        Run demo on trained arrhythmia model using the same configuration file.

        ```bash
        heartkit -m demo -t arrhythmia -c ./configs/arrhythmia-class-2.json
        ```

## <span class="sk-h2-span">Use HeartKit with Python</span>

__HeartKit__ Python package allows for more fine-grained control and customization. You can use the package to train, evaluate, and deploy models for a variety of tasks. The package is designed to be simple and easy to use.

For example, you can create a custom model, train it, evaluate its performance on a validation set, and even export a quantized TensorFlow Lite model for deployment. Check out the [Python Guide](./usage/python.md) to learn more about using HeartKit as a Python package.

!!! Example

    ```python
    import heartkit as hk

    ds_params = hk.HKDownloadParams.parse_file("download-datasets.json")

    with open("configuration.json", "r", encoding="utf-8") as file:
        config = json.load(file)

    train_params = hk.HKTrainParams.model_validate(config)
    test_params = hk.HKTestParams.model_validate(config)
    export_params = hk.HKExportParams.model_validate(config)

    # Download datasets
    hk.datasets.download_datasets(ds_params)

    task = hk.TaskFactory.get("arrhythmia")

    # Train arrhythmia model
    task.train(train_params)

    # Evaluate arrhythmia model
    task.evaluate(test_params)

    # Export arrhythmia model
    task.export(export_params)

    ```

!!! note
    If using editable mode via Poetry, be sure to activate the python environment: `poetry shell`. On Windows using Powershell, use `.venv\Scripts\activate.ps1`.

---
