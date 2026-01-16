# Quickstart
<!-- # :octicons-heart-fill-24:{ .heart } Overview -->

## Install heartKIT

We provide several installation methods including pip, uv, and Docker. Install __heartKIT__ via pip/uv for the latest stable release or by cloning the GitHub repo for the most up-to-date. Additionally, a [VSCode Dev Container](https://code.visualstudio.com/docs/devcontainers/containers) is available and defined in [./.devcontainer](https://github.com/AmbiqAI/heartkit/tree/main/.devcontainer) to run in an isolated Docker environment.

!!! install

    === "Git clone"

        Clone the repository if you are interested in contributing to the development or wish to experiment with the latest source code. After cloning, navigate into the directory and install the package. In this mode, uv is recommended.

        ```bash
        # Clone the repository
        git clone https://github.com/AmbiqAI/heartkit.git

        # Navigate to the cloned directory
        cd heartkit

        # Install the package in editable mode for development
        uv sync
        ```

        When using editable mode via uv, be sure to activate the python environment: `source .venv/bin/activate`. <br>
        On Windows using Powershell, use `.venv\Scripts\activate`.

    === "PyPI install"

        Install the heartKIT package using pip or uv.
        Visit the Python Package Index (PyPI) for more details on the package: [https://pypi.org/project/heartkit/](https://pypi.org/project/heartkit/)

        ```bash
        # Install with pip
        pip install heartkit
        ```

        Or, if you prefer to use uv, you can install the package with the following command:

        ```bash
        # Install with uv
        uv add heartkit
        ```

        Alternatively, you can install the latest development version directly from the GitHub repository. Make sure to have the Git command-line tool installed on your system. The @main command installs the main branch and may be modified to another branch, i.e. @canary.

        ```bash
        pip install git+https://github.com/AmbiqAI/heartkit.git@main
        ```

        Or, using uv:

        ```bash
        uv add git+https://github.com/AmbiqAI/heartkit.git@main
        ```


## Requirements

* [Python ^3.11+](https://www.python.org)
* [uv ^0.7.10+](https://docs.astral.sh/uv/getting-started/installation/)

Check the project's [pyproject.toml](https://github.com/AmbiqAI/heartkit/blob/main/pyproject.toml) file for a list of up-to-date Python dependencies. Note that the installation methods above install all required dependencies. The following are optional dependencies only needed when running `demo` command using Ambiq's evaluation board (`EVB`) backend:

* [Arm GNU Toolchain ^12.2](https://developer.arm.com/downloads/-/arm-gnu-toolchain-downloads)
* [Segger J-Link ^7.92](https://www.segger.com/downloads/jlink/)

Once installed, __heartKIT__ can be used as either a CLI-based tool or as a Python package to perform advanced experimentation.

---

## Use heartKIT with CLI

The heartKIT command line interface (CLI) allows for simple single-line commands to download datasets, train models, evaluate performance, and export models. The CLI requires no customization or Python code. You can simply run all the built-in tasks from the terminal with the __heartkit__ command. Check out the [CLI Guide](./usage/cli.md) to learn more about available options.

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

## Use heartKIT with Python

The __heartKIT__ Python package allows for more fine-grained control and customization. You can use the package to train, evaluate, and deploy models for a variety of tasks. You can create custom datasets, models, and tasks and register them with corresponding factories and use them like built-in tasks.

For example, you can create a custom task, train it, evaluate its performance on a validation set, and even export a quantized TensorFlow Lite model for deployment. Check out the [Python Guide](./usage/python.md) to learn more about using heartKIT as a Python package.

!!! Example

    ```py linenums="1"

    import heartkit as hk

    params = hk.HKTaskParams(...)  # Expand to see example (1)

    task = hk.TaskFactory.get("rhythm")

    task.download(params)  # Download dataset(s)

    task.train(params)  # Train the model

    task.evaluate(params)  # Evaluate the model

    task.export(params)  # Export to TFLite

    ```

    1. Configuration parameters:
    --8<-- "assets/usage/python-configuration.md"


---
