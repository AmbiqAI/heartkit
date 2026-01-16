#

[![](./assets/heartkit-logo-light.png#only-light)](https://ambiqai.github.io/heartkit/)
[![](./assets/heartkit-logo-dark.png#only-dark)](https://ambiqai.github.io/heartkit/)

*An AI Development Kit for real-time heart-monitoring on ultra-low power SoCs*

## Overview

Introducing heartKIT, an AI Development Kit (ADK) that enables developers to easily train and deploy real-time __heart-monitoring__ models onto [Ambiq's family of ultra-low power SoCs](https://ambiq.com/soc/). The kit provides a variety of datasets, efficient model architectures, and heart-related tasks out of the box. In addition, heartKIT provides optimization and deployment routines to generate efficient inference models. Finally, the kit includes a number of pre-trained models and task-level demos to showcase the capabilities.

**Key Features:**

* **Real-time**: Inference is performed in real-time on battery-powered, edge devices.
* **Efficient**: Leverage Ambiq's ultra low-power SoCs for extreme energy efficiency.
* **Extensible**: Easily add new tasks, models, and datasets to the framework.
* **Open Source**: heartKIT is open source and available on GitHub.

Please explore the heartKIT Docs, a comprehensive resource designed to help you understand and utilize all the built-in features and capabilities.

## Getting Started

- **Install** `heartKIT` with pip/uv and getting up and running in minutes. &nbsp; [:material-clock-fast: Install heartKIT](./quickstart.md/#install-heartkit){ .md-button }
- **Train** a model with a custom network &nbsp; [:fontawesome-solid-brain: Train a Model](modes/train.md){ .md-button }
- **Tasks** `heartKIT` provides tasks like rhythm, segment, and denoising &nbsp; [:material-magnify-expand: Explore Tasks](tasks/index.md){ .md-button }
- **Datasets** Several built-in datasets can be leveraged &nbsp; [:material-database-outline: Explore Datasets](./datasets/index.md){ .md-button }
- **Model Zoo** Pre-trained models are available for each task &nbsp; [:material-download: Explore Models](./zoo/index.md){ .md-button }
- **Guides** Detailed guides on tasks, models, and datasets &nbsp; [:material-book-open-page-variant: Explore Guides](./guides/index.md){ .md-button }

## Installation

To get started, first install the python package `heartkit` along with its dependencies via `Git` or `PyPi`:

=== "PyPI install"
    <br/>
    <div class="termy">

    ```console
    $ pip install heartkit

    ---> 100%
    ```

    </div>

=== "Git clone"
    <br/>
    <div class="termy">

    ```console
    $ git clone https://github.com/AmbiqAI/heartkit.git
    Cloning into 'heartkit'...
    Resolving deltas: 100% (3491/3491), done.
    $ cd heartkit
    $ uv install

    ---> 100%
    ```

    </div>

---

## Usage

__heartKIT__ can be used as either a CLI-based tool or as a Python package to perform advanced development. In both forms, heartKIT exposes a number of modes and tasks outlined below. In addition, by leveraging highly-customizable configurations and extendable factories, heartKIT can be used to create custom workflows for a given application with minimal coding. Refer to the [Quickstart](./quickstart.md) to quickly get up and running in minutes.

---

## [Tasks](./tasks/index.md)

__heartKIT__ includes a number of built-in [tasks](./tasks/index.md). Each task provides reference routines for training, evaluating, and exporting the model. The routines can be customized by providing highly flexibile configuration files/objects. Additionally, new tasks can be added to the __heartKIT__ framework by defining a new [Task class](./tasks/byot.md) and registering it to the [__Task Factory__](./tasks/byot.md).

- **[Denoise](./tasks/denoise.md)**: Remove noise and artifacts from signals
- **[Segmentation](./tasks/segmentation.md)**: Perform ECG/PPG based segmentation
- **[Rhythm](./tasks/rhythm.md)**: Heart rhythm classification (AFIB, AFL)
- **[Beat](./tasks/beat.md)**: Beat-level classification (NORM, PAC, PVC, NOISE)
<!-- - **[Diagnostic](./tasks/diagnostics.md)**: Diagnostic classification (MI, STTC, LVH) -->
- **[Bring-Your-Own-Task (BYOT)](./tasks/byot.md)**: Create and register custom tasks

---

## [Modes](./modes/index.md)

__heartKIT__ provides a number of [modes](./modes/index.md) that can be invoked for a given task. These modes can be accessed via the CLI or directly from a [Task](./tasks/index.md). Each mode is accompanied by a set of [task parameters](./modes/configuration.md#hktaskparams) that can be customized to fit the user's needs.

- **[Download](./modes/download.md)**: Download specified datasets
- **[Train](./modes/train.md)**: Train a model for specified task and datasets
- **[Evaluate](./modes/evaluate.md)**: Evaluate a model for specified task and datasets
- **[Export](./modes/export.md)**: Export a trained model to TensorFlow Lite and TFLM
- **[Demo](./modes/demo.md)**: Run task-level demo on PC or remotely on Ambiq EVB

---

## [Datasets](./datasets/index.md)

The ADK includes several built-in [datasets](./datasets/index.md) for training __heart-monitoring__ related tasks. We also provide synthetic dataset generators for signals such as ECG, PPG, and RSP along with segmentation and fiducials. Each included dataset inherits from [HKDataset](api/heartkit/datasets/dataset.md) that provides consistent interface for downloading and accessing the data. Additional datasets can be added to the heartKIT framework by creating a new dataset class and registering it to the dataset factory, [DatasetFactory](./models/index.md#model-factory). Check out the [Datasets Guide](./datasets/index.md) to learn more about the available datasets along with their corresponding licenses and limitations.

* **[Icentia11k](./datasets/icentia11k.md)**: 11-lead ECG data collected from 11,000 subjects captured continously over two weeks.
* **[LUDB](./datasets/ludb.md)**: 200 ten-second 12-lead ECG records w/ annotated P-wave, QRS, and T-wave boundaries.
* **[QTDB](./datasets/qtdb.md)**: 100+ fifteen-minute two-lead ECG recordings w/ annotated P-wave, QRS, and T-wave boundaries.
* **[LSAD](./datasets/lsad.md)**: 10-second, 12-lead ECG dataset collected from 45,152 subjects w/ over 100 scp codes.
* **[PTB-XL](./datasets/ptbxl.md)**: 10-second, 12-lead ECG dataset collected from 18,885 subjects w/ 72 different diagnostic classes.
* **[Synthetic](./datasets/synthetic.md)**: A synthetic dataset generator provided by [physioKIT](https://ambiqai.github.io/physiokit).
* **[Bring-Your-Own-Dataset (BYOD)](./datasets/byod.md)**: Add and register new datasets to the framework.

---

## [Models](./models/index.md)

__heartKIT__ provides a variety of model architectures geared towards efficient, real-time edge applications. These models are provided by Ambiq's [helia-edge](https://ambiqai.github.io/helia-edge/) and expose a set of parameters that can be used to fully customize the network for a given application. In addition, heartKIT includes a model factory, [ModelFactory](./models/index.md#model-factory), to register current models as well as allow new custom architectures to be added. Check out the [Models Guide](./models/index.md) to learn more about the available network architectures and model factory.

- **[TCN](https://ambiqai.github.io/helia-edge/api/helia_edge/models/tcn)**: A CNN leveraging dilated convolutions (key=`tcn`)
- **[U-Net](https://ambiqai.github.io/helia-edge/api/helia_edge/models/unet)**: A CNN with encoder-decoder architecture for segmentation tasks (key=`unet`)
- **[U-NeXt](https://ambiqai.github.io/helia-edge/api/helia_edge/models/unext)**: A U-Net variant leveraging MBConv blocks (key=`unext`)
- **[EfficientNetV2](https://ambiqai.github.io/helia-edge/api/helia_edge/models/efficientnet)**: A CNN leveraging MBConv blocks (key=`efficientnet`)
- **[MobileOne](https://ambiqai.github.io/helia-edge/api/helia_edge/models/mobileone)**: A CNN aimed at sub-1ms inference (key=`mobileone`)
- **[ResNet](https://ambiqai.github.io/helia-edge/api/helia_edge/models/resnet)**: A popular CNN often used for vision tasks (key=`resnet`)
- **[Conformer](https://ambiqai.github.io/helia-edge/api/helia_edge/models/conformer)**: A transformer composed of both convolutional and self-attention blocks (key=`conformer`)
- **[MetaFormer](https://ambiqai.github.io/helia-edge/api/helia_edge/models/metaformer)**: A transformer composed of both spatial mixing and channel mixing blocks (key=`metaformer`)
- **[TSMixer](https://ambiqai.github.io/helia-edge/api/helia_edge/models/tsmixer)**: An All-MLP Architecture for Time Series Classification (key=`tsmixer`)
- **[Bring-Your-Own-Model (BYOM)](./models/byom.md)**: Register new SoTA model architectures w/ custom configurations

---

## [Model Zoo](./zoo/index.md)

The ADK includes a number of pre-trained models and configurationn recipes for the built-in tasks. These models are trained on a variety of datasets and are optimized for deployment on Ambiq's ultra-low power SoCs. In addition to providing links to download the models, __heartKIT__ provides the corresponding configuration files and performance metrics. The configuration files allow you to easily recreate the models or use them as a starting point for custom solutions. Furthermore, the performance metrics provide insights into the trade-offs between model complexity and performance. Check out the [Model Zoo](./zoo/index.md) to learn more about the available models and their corresponding performance metrics.

---

## [Guides](./guides/index.md)

Checkout the [Guides](./guides/index.md) to see detailed examples and tutorials on how to use heartKIT for a variety of tasks. The guides provide step-by-step instructions on how to train, evaluate, and deploy models for a given task. In addition, the guides provide insights into the design choices and performance metrics for the models. The guides are designed to help you get up and running quickly and to provide a deeper understanding of the capabilities provided by heartKIT.

---
