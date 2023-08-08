<p align="center">
  <a href="https://github.com/AmbiqAI/heartkit"><img src="./docs/assets/heartkit-banner.png" alt="HeartKit"></a>
</p>

---

**Documentation**: <a href="https://ambiqai.github.io/heartkit" target="_blank">https://ambiqai.github.io/heartkit</a>

**Source Code**: <a href="https://github.com/AmbiqAI/heartkit" target="_blank">https://github.com/AmbiqAI/heartkit</a>

---

HeartKit is an optimized open-source TinyML model purpose-built to enable running a variety of real-time heart-monitoring applications on battery-powered, edge devices. By leveraging a modern multi-head network architecture coupled with Ambiq's ultra low-power SoC, the model is designed to be **efficient**, **explainable**, and **extensible**.

The architecture consists of an **ECG segmentation** model followed by three upstream heads: **HRV head**, **arrhythmia head**, and **beat head**. The ECG segmentation model serves as the backbone and is used to annotate every sample as either P-wave, QRS, T-wave, or none. The arrhythmia head is used to detect the presence of Atrial Fibrillation (AFIB) or Atrial Flutter (AFL). The HRV head is used to calculate heart rate, rhythm (e.g., bradycardia), and heart rate variability from the R peaks. Lastly, the beat head is used to identify individual irregular beats (PAC, PVC).

**Key Features:**

* **Efficient**: Novel architecture coupled w/ Ambiq's ultra low-power SoCs enable extreme energy efficiency.
* **Explainable**: Inference results are paired with metrics to provide explainable insights.
* **Extensible**: Add or remove heads for desired end application.

## Requirements

* [Python 3.11+](https://www.python.org)
* [Poetry 1.2.1+](https://python-poetry.org/docs/#installation)

The following are also required to compile/flash the binary for the EVB demo:

* [Arm GNU Toolchain 11.3](https://developer.arm.com/downloads/-/arm-gnu-toolchain-downloads)
* [Segger J-Link v7.56+](https://www.segger.com/downloads/jlink/)

> NOTE: A [VSCode Dev Container](https://code.visualstudio.com/docs/devcontainers/containers) is also available and defined in `./.devcontainer`.


## Installation

To get started, first install the local python package `heartkit` along with its dependencies via `Poetry`:

```bash
poetry install
```

## Usage

__HeartKit__ is intended to be used as either a CLI-based app or as a python package to perform additional tasks and experiments.

### Modes:

* `download`: Download datasets
* `train`: Train a model for specified task and dataset(s)
* `evaluate`: Evaluate a model for specified task and dataset(s)
* `export`: Export a trained model to TensorFlow Lite and TFLM.
* `demo`: Run full demo on PC or EVB

### Tasks:

* `Segmentation`: Perform ECG based segmentation (P-Wave, QRS, T-Wave)
* `HRV`: Heart rate, rhythm, HRV metrics (RR interval)
* `Arrhythmia`: Heart arrhythmia detection (AFIB, AFL)
* `Beat`: Classify individual beats (PAC, PVC)


### Using CLI

The CLI provides a number of commands discussed below. In general, reference configurations are provided to download datasets, train/evaluate/export models, and lastly demo model(s) on PC or Apollo 4 EVB. Pre-trained reference models are also included to enable running inference and the demo immediately.

```bash
heartkit
--task [segmentation, arrhythmia, beat, hrv]
--mode [download, train, evaluate, export, demo]
--config ["./path/to/config.json", or '{"raw: "json"}']
```

> NOTE: Before running commands, be sure to activate python environment: `poetry shell`. On Windows using Powershell, use `.venv\Scripts\activate.ps1`.

#### __1. Download Datasets__

The `download` command is used to download all datasets specified in the configuration file. Please refer to [Datasets section](#datasets) for details on the available datasets.

The following command will download and prepare all currently used datasets.

```bash
heartkit --mode download --config ./configs/download-datasets.json
```

> NOTE: The __Icentia11k dataset__ requires roughly 200 GB of disk space and can take around 2 hours to download.

#### __2. Train Model__

The `train` command is used to train a HeartKit model. The following command will train the arrhythmia model using the reference configuration. Please refer to `heartkit/defines.py` to see supported options.

```bash
heartkit --task arrhythmia --mode train --config ./configs/train-arrhythmia-model.json
```

#### __3. Evaluate Model__

The `evaluate` command will evaluate the performance of the model on the reserved test set. A confidence threshold can also be set such that a label is only assigned when the model's probability is greater than the threshold; otherwise, a label of inconclusive will be assigned.

```bash
heartkit --task arrhythmia --mode evaluate --config ./configs/test-arrhythmia-model.json
```

#### __4. Export Model__

The `export` command will convert the trained TensorFlow model into both TFLite (TFL) and TFLite for microcontroller (TFLM) variants. The command will also verify the models' outputs match. Post-training quantization can also be enabled by setting the `quantization` flag in the configuration.

```bash
heartkit --task arrhythmia --mode export --config ./configs/export-arrhythmia-model.json
```

Once converted, the TFLM header file will be copied to location specified by `tflm_file`. If parameters were changed (e.g. window size, quantization), `./evb/src/constants.h` will need to be updated.

#### __5. Demo__

The `demo` command is used to run a full-fledged HeartKit demonstration. The demo is decoupled into three tasks: (1) a REST server to provide a unified API, (2) a front-end UI, and (3) a backend to fetch samples and perform inference. The host PC performs tasks (1) and (2). For (3), the trained models can run on either the `PC` or an Apollo 4 evaluation board (`EVB`) by setting the `backend` field in the configuration. When the `PC` backend is selected, the host PC will perform task (3) entirely to fetch samples and perform inference. When the `EVB` backend is selected, the `EVB` will perform inference using either sensor data or prior data. The PC connects to the `EVB` via RPC over serial transport to provide sample data and capture inference results.

Please refer to [Arrhythmia demo tutorial](./docs/tutorials/arrhythmia-demo.md) and [HeartKit demo tutorial](./docs/tutorials/heartkit-demo.md) for further instructions.

## Model Architecture

HeartKit leverages a multi-head network- a backbone segmentation model followed by 3 uptream heads:

* __Segmentation backbone__ utilizes a custom 1-D UNET architecture to perform ECG segmentation.
* __HRV head__ utilizes segmentation results to derive a number of useful metrics including heart rate, rhythm and RR interval.
* __Arrhythmia head__ utilizes a 1-D MBConv CNN to detect arrhythmias include AFIB and AFL.
* __Beat-level head__ utilizes a 1-D MBConv CNN to detect irregular individual beats (PAC, PVC).

![](./docs/assets/heartkit-architecture.svg)

## Datasets

HeartKit leverages several open-source datasets for training each of the HeartKit models. Additionally, HeartKit contains a customizable synthetic 12-lead ECG generator. Check out the [Datasets Guide](./datasets.md) to learn more about the datasets used along with their corresponding licenses and limitations.

## Results

The following table provides the latest performance and accuracy results of all models when running on Apollo4 Plus EVB.

| Task           | Params   | FLOPS   | Metric      |
| -------------- | -------- | ------- | ----------- |
| Segmentation   | 28K      | 5.7M    | 91.9% IOU   |
| Arrhythmia     | 49K      | 3.5M    | 99.3% F1    |
| Beat           | 72K      | 2.1M    | 91.6% F1    |
| HRV            | N/A      | N/A     | N/A         |


## References

* [ECG Heartbeat classification using deep transfer learning with Convolutional Neural Network and STFT technique](https://arxiv.org/abs/2206.14200)
* [Classification of ECG based on Hybrid Features using CNNs for Wearable Applications](https://arxiv.org/pdf/2206.07648.pdf)
* [ECG Heartbeat classification using deep transfer learning with Convolutional Neural Network and STFT technique](https://arxiv.org/pdf/2206.14200.pdf)
* [U2-Net: Going Deeper with Nested U-Structure for Salient Object Detection](https://arxiv.org/abs/2005.09007)
* [UNET 3+: A FULL-SCALE CONNECTED UNET FOR MEDICAL IMAGE SEGMENTATION](https://arxiv.org/pdf/2004.08790.pdf)
* [ResUNet-a: a deep learning framework for semantic segmentation of remotely sensed data](https://arxiv.org/pdf/1904.00592.pdf)
