# Home

<p align="center">
  <a href="https://github.com/AmbiqAI/heartkit"><img src="./assets/heartkit-banner.png" alt="HeartKit"></a>
</p>

---

**Documentation**: <a href="https://ambiqai.github.io/heartkit" target="_blank">https://ambiqai.github.io/heartkit</a>

**Source Code**: <a href="https://github.com/AmbiqAI/heartkit" target="_blank">https://github.com/AmbiqAI/heartkit</a>

---

Introducing HeartKit, an AI Development Kit (ADK) that enables developers to easily train and deploy real-time __heart-monitoring__ models onto [Ambiq's family of ultra-low power SoCs](https://ambiq.com/soc/). The kit provides a variety of datasets, efficient model architectures, and heart-related tasks. In addition, HeartKit provides optimization and deployment routines to generate efficient inference models. Finally, the kit includes a number of pre-trained models and task-level demos to showcase the capabilities.

**Key Features:**

* **Real-time**: Inference is performed in real-time on battery-powered, edge devices.
* **Efficient**: Leverage Ambiq's ultra low-power SoCs for extreme energy efficiency.
* **Extensible**: Easily add new tasks, models, and datasets to the framework.
* **Open Source**: HeartKit is open source and available on GitHub.

Please explore the HeartKit Docs, a comprehensive resource designed to help you understand and utilize all the built-in features and capabilities.

## <span class="sk-h2-span">Getting Started</span>

- **Install** `HeartKit` with pip/poetry and getting up and running in minutes. &nbsp; [:material-clock-fast: Install HeartKit](./quickstart.md/#install){ .md-button }
- **Train** a model with a custom network &nbsp; [:fontawesome-solid-brain: Train a Model](modes/train.md){ .md-button }
- **Tasks** `HeartKit` provides tasks like rhythm, segment, and denoising &nbsp; [:material-magnify-expand: Explore Tasks](tasks/index.md){ .md-button }
- **Datasets** Several built-in datasets can be leveraged &nbsp; [:material-database-outline: Explore Datasets](./datasets/index.md){ .md-button }
- **Model Zoo** Pre-trained models are available for each task &nbsp; [:material-download: Explore Models](./zoo/index.md){ .md-button }

## <span class="sk-h2-span">Installation</span>

To get started, first install the local python package `heartkit` along with its dependencies via `pip` or `Poetry`:

=== "Poetry install"

    <div class="termy">

    ```console
    $ poetry install .

    ---> 100%
    ```

    </div>

=== "Pip install"

    <div class="termy">

    ```console
    $ pip install heartkit

    ---> 100%
    ```

    </div>

---

## <span class="sk-h2-span">Usage</span>

__HeartKit__ can be used as either a CLI-based tool or as a Python package to perform advanced development. In both forms, HeartKit exposes a number of modes and tasks outlined below. In addition, by leveraging highly-customizable configurations, HeartKit can be used to create custom workflows for a given application with minimal coding. Refer to the [Quickstart](./quickstart.md) to quickly get up and running in minutes.

---

## <span class="sk-h2-span">Modes</span>

__HeartKit__ provides a number of [modes](./modes/index.md) that can be invoked for a given task. These modes can be accessed via the CLI or directly from the `task` within the Python package.

- **[Download](./modes/download.md)**: Download specified datasets
- **[Train](./modes/train.md)**: Train a model for specified task and datasets
- **[Evaluate](./modes/evaluate.md)**: Evaluate a model for specified task and datasets
- **[Export](./modes/export.md)**: Export a trained model to TensorFlow Lite and TFLM
- **[Demo](./modes/demo.md)**: Run task-level demo on PC or remotely on Ambiq EVB

---

## <span class="sk-h2-span">Task Factory</span>

__HeartKit__ includes a number of built-in [tasks](./tasks/index.md). Each task provides reference routines for training, evaluating, and exporting the model. The routines can be customized by providing a configuration file or by setting the parameters directly in the code. Additional tasks can be easily added to the __HeartKit__ framework by creating a new task class and registering it to the __task factory__.

- **[Denoise](./tasks/denoise.md)**: Denoise ECG signal
- **[Segmentation](./tasks/segmentation.md)**: Perform ECG based segmentation (P-Wave, QRS, T-Wave)
- **[Rhythm](./tasks/rhythm.md)**: Heart rhythm classification (AFIB, AFL)
- **[Beat](./tasks/beat.md)**: Beat-level classification (NORM, PAC, PVC, NOISE)
<!-- - **[Diagnostic](./tasks/diagnostics.md)**: Diagnostic classification (MI, STTC, LVH) -->
- **[BYOT](./tasks/byot.md)**: Bring-Your-Own-Task (BYOT) to create custom tasks

---

## <span class="sk-h2-span">Model Factory</span>

__HeartKit__ provides a __model factory__ that allows you to easily create and train customized models. The model factory includes a number of modern networks well suited for efficient, real-time edge applications. Each model architecture exposes a number of high-level parameters that can be used to customize the network for a given application. These parameters can be set as part of the configuration accessible via the CLI and Python package. Check out the [Model Factory Guide](./models/index.md) to learn more about the available network architectures.

---

## <span class="sk-h2-span">Dataset Factory</span>

__HeartKit__ exposes several open-source datasets for training each of the HeartKit tasks via the __dataset factory__. For certain tasks, we also provide synthetic data provided by [PhysioKit](https://ambiqai.github.io/physiokit) to help improve model generalization. Each dataset has a corresponding Python class to aid in downloading and generating data for the given task. Additional datasets can be added to the HeartKit framework by creating a new dataset class and registering it to the dataset factory. Check out the [Dataset Factory Guide](./datasets/index.md) to learn more about the available datasets along with their corresponding licenses and limitations.


* **[Icentia11k](./datasets/icentia11k.md)**: 11-lead ECG data collected from 11,000 subjects captured continously over two weeks.
* **[LUDB](./datasets/ludb.md)**: 200 ten-second 12-lead ECG records w/ annotated P-wave, QRS, and T-wave boundaries.
* **[QTDB](./datasets/qtdb.md)**: 100+ fifteen-minute two-lead ECG recordings w/ annotated P-wave, QRS, and T-wave boundaries.
* **[LSAD](./datasets/lsad.md)**: 10-second, 12-lead ECG dataset collected from 45,152 subjects w/ over 100 scp codes.
* **[PTB-XL](./datasets/ptbxl.md)**: 10-second, 12-lead ECG dataset collected from 18,885 subjects w/ 72 different diagnostic classes.
* **[Synthetic](./datasets/synthetic.md)**: A synthetic dataset generator provided by [PhysioKit](https://ambiqai.github.io/physiokit).
* **[BYOD](./datasets/byod.md)**: Bring-Your-Own-Dataset (BYOD) to add additional datasets.

---

## <span class="sk-h2-span">Model Zoo</span>

A number of pre-trained models are available for each task. These models are trained on a variety of datasets and are optimized for deployment on Ambiq's ultra-low power SoCs. In addition to providing links to download the models, __HeartKit__ provides the corresponding configuration files and performance metrics. The configuration files allow you to easily recreate the models or use them as a starting point for custom solutions. Furthermore, the performance metrics provide insights into the model's accuracy, precision, recall, and F1 score. For a number of the models, we provide experimental and ablation studies to showcase the impact of various design choices. Check out the [Model Zoo](./zoo/index.md) to learn more about the available models and their corresponding performance metrics. Also explore the [Experiments](./experiments/index.md) to learn more about the ablation studies and experimental results.

---
