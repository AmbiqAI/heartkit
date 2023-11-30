---
title:
---
#

<p align="center">
  <a href="https://github.com/AmbiqAI/heartkit"><img src="./assets/heartkit-banner.png" alt="HeartKit"></a>
</p>

---

**Documentation**: <a href="https://ambiqai.github.io/heartkit" target="_blank">https://ambiqai.github.io/heartkit</a>

**Source Code**: <a href="https://github.com/AmbiqAI/heartkit" target="_blank">https://github.com/AmbiqAI/heartkit</a>

---

HeartKit is an optimized open-source TinyML model purpose-built to enable running a variety of real-time heart-monitoring applications on battery-powered, edge devices.

...

**Key Features:**

* **Efficient**: Novel architecture coupled w/ Ambiq's ultra low-power SoCs enable extreme energy efficiency.
* **Explainable**: Inference results are paired with metrics to provide explainable insights.
* **Extensible**: Add or remove heads for desired end application.

## Requirements

* [Python ^3.11+](https://www.python.org)
* [Poetry ^1.6.1+](https://python-poetry.org/docs/#installation)

The following are also required to compile/flash the binary for the EVB demo:

* [Arm GNU Toolchain ^12.2](https://developer.arm.com/downloads/-/arm-gnu-toolchain-downloads)
* [Segger J-Link ^7.92](https://www.segger.com/downloads/jlink/)

!!! note
    A [VSCode Dev Container](https://code.visualstudio.com/docs/devcontainers/containers) is also available and defined in [./.devcontainer](https://github.com/AmbiqAI/heartkit/tree/main/.devcontainer).

## Installation

To get started, first install the local python package `heartkit` along with its dependencies via `Poetry`:

<div class="termy">

```console
$ poetry install

---> 100%
```
</div>

---

## Usage

__HeartKit__ can be used as either a CLI-based app or as a python package to perform advanced experimentation. In both forms, HeartKit exposes a number of modes and tasks discussed below. Refer to the [Overview Guide](./overview.md) to learn more about available options and configurations.

---

## Modes

* `download`: Download datasets
* `train`: Train a model for specified task and dataset(s)
* `evaluate`: Evaluate a model for specified task and dataset(s)
* `export`: Export a trained model to TensorFlow Lite and TFLM
* `demo`: Run task-level demo on PC or EVB

---

## Tasks

* `Segmentation`: Perform ECG based segmentation (P-Wave, QRS, T-Wave)
* `Arrhythmia`: Heart arrhythmia detection (AFIB, AFL)
* `Beat`: Classify individual beats (NORM, PAC, PVC, NOISE)

---

## Architecture

HeartKit leverages modern architectural design strategies to achieve high accuracy while maintaining a small memory footprint and low power consumption.

* Seperable (depthwise + pointwise) Convolutions
* Inverted Residual Bottlenecks
* Squeeze & Excitation Blocks
* MBConv Blocks
* Over-Parameterized Convolutional Branches
* Dilated Convolutions

Refer to specific task guides for additional details on the full model design.

---

## Sensing Modalities

The two primary sensing modalities to monitor cardiac cycles are electrocardiograph (ECG) and photoplethysmography (PPG).

## Datasets

HeartKit leverages several open-source datasets for training each of the HeartKit models. Additionally, HeartKit leverages [PhysioKit's synthetic ECG generator](https://ambiqai.github.io/physiokit) to generate additional training data. Check out the [Datasets Guide](./datasets.md) to learn more about the datasets used along with their corresponding licenses and limitations.

---

## Results

The following table provides the latest performance and accuracy results of all pre-trained task models when running on Apollo4 Plus EVB. Additional result details can be found in [Results Section](./results.md) alogn with task-level documentation.

| Task           | Params   | FLOPS   | Metric     | Cycles/Inf | Time/Inf   |
| -------------- | -------- | ------- | ---------- | ---------- | ---------- |
| Segmentation   | 33K      | 6.5M    | 87.0% IOU  | 531ms      | 102M       |
| Arrhythmia     | 50K      | 3.6M    | 99.0% F1   | 465ms      | 89M        |
| Beat           | 73K      | 2.2M    | 91.5% F1   | 241ms      | 46M        |

---

## References

* [ECG Heartbeat classification using deep transfer learning with Convolutional Neural Network and STFT technique](https://arxiv.org/abs/2206.14200)
* [Classification of ECG based on Hybrid Features using CNNs for Wearable Applications](https://arxiv.org/pdf/2206.07648.pdf)
* [ECG Heartbeat classification using deep transfer learning with Convolutional Neural Network and STFT technique](https://arxiv.org/pdf/2206.14200.pdf)
* [U2-Net: Going Deeper with Nested U-Structure for Salient Object Detection](https://arxiv.org/abs/2005.09007)
* [UNET 3+: A FULL-SCALE CONNECTED UNET FOR MEDICAL IMAGE SEGMENTATION](https://arxiv.org/pdf/2004.08790.pdf)
* [ResUNet-a: a deep learning framework for semantic segmentation of remotely sensed data](https://arxiv.org/pdf/1904.00592.pdf)
