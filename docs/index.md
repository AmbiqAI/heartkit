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

HeartKit is an optimized open-source TinyML model purpose-built to enable running a variety of real-time heart-monitoring applications on battery-powered, edge devices. By leveraging a modern multi-head network architecture coupled with Ambiq's ultra low-power SoC, the model is designed to be **efficient**, **explainable**, and **extensible**.

The architecture consists of an **ECG segmentation** model followed by three upstream heads: **HRV head**, **arrhythmia head**, and **beat head**. The ECG segmentation model serves as the backbone and is used to annotate every sample as either P-wave, QRS, T-wave, or none. The arrhythmia head is used to detect the presence of Atrial Fibrillation (AFIB) or Atrial Flutter (AFL). The HRV head is used to calculate heart rate, rhythm (e.g., bradycardia), and heart rate variability from the R peaks. Lastly, the beat head is used to identify individual irregular beats (PAC, PVC).

**Key Features:**

* **Efficient**: Novel architecture coupled w/ Ambiq's ultra low-power SoCs enable extreme energy efficiency.
* **Explainable**: Inference results are paired with metrics to provide explainable insights.
* **Extensible**: Add or remove heads for desired end application.

## Requirements

* [Python 3.10+](https://www.python.org)
* [Poetry 1.2.1+](https://python-poetry.org/docs/#installation)

The following are also required to compile/flash the binary for the EVB demo:

* [Arm GNU Toolchain 11.3](https://developer.arm.com/downloads/-/arm-gnu-toolchain-downloads)
* [Segger J-Link v7.56+](https://www.segger.com/downloads/jlink/)

!!! note
    A [VSCode Dev Container](https://code.visualstudio.com/docs/devcontainers/containers) is also available and defined in [./.devcontainer](https://github.com/AmbiqAI/heartkit/tree/main/.devcontainer).

## Installation

<div class="termy">

```console
$ poetry install

---> 100%
```
</div>


## Usage

__HeartKit__ can be used as either a CLI-based app or as a python package to perform advanced experimentation. In both forms, HeartKit exposes a number of modes and tasks discussed below. Refer to the [Overview Guide](./overview.md) to learn more about available options and configurations.

## Modes

* `download`: Download datasets
* `train`: Train a model for specified task and dataset(s)
* `evaluate`: Evaluate a model for specified task and dataset(s)
* `export`: Export a trained model to TensorFlow Lite and TFLM
* `demo`: Run full demo on PC or EVB

## Tasks

* `Segmentation`: Perform ECG based segmentation (P-Wave, QRS, T-Wave)
* `HRV`: Heart rate, rhythm, HRV metrics (RR interval)
* `Arrhythmia`: Heart arrhythmia detection (AFIB, AFL)
* `Beat`: Classify individual beats (PAC, PVC)

****
## Architecture

HeartKit leverages a multi-head network- a backbone segmentation model followed by 3 uptream heads:

* __Segmentation backbone__ utilizes a custom 1-D UNET architecture to perform ECG segmentation.
* __HRV head__ utilizes segmentation results to derive a number of useful metrics including heart rate, rhythm and RR interval.
* __Arrhythmia head__ utilizes a 1-D MBConv CNN to detect arrhythmias include AFIB and AFL.
* __Beat-level head__ utilizes a 1-D MBConv CNN to detect irregular individual beats (PAC, PVC).

<p align="center">
  <img src="./assets/heartkit-architecture.svg" alt="HeartKit Architecture">
</p>

Refer to [Architecture Overview](./architecture.md) for additional details on the model design.


## Datasets

HeartKit leverages several open-source datasets for training each of the HeartKit models. Additionally, HeartKit contains a customizable synthetic 12-lead ECG generator. Check out the [Datasets Guide](./datasets.md) to learn more about the datasets used along with their corresponding licenses and limitations.


## Results

The following table provides the latest performance and accuracy results of all models when running on Apollo4 Plus EVB. Additional result details can be found in [Results Section](./results.md).

| Task           | Params   | FLOPS   | Metric      |
| -------------- | -------- | ------- | ----------- |
| Segmentation   | 105K     | 19.3M   | IOU=85.3%   |
| Arrhythmia     | 76K      | 7.2M    | F1=99.4%    |
| Beat           | 79K      | 1.6M    | F1=91.6%    |
| HRV            | N/A      | N/A     | N/A         |

## References

* [ECG Heartbeat classification using deep transfer learning with Convolutional Neural Network and STFT technique](https://arxiv.org/abs/2206.14200)
* [Classification of ECG based on Hybrid Features using CNNs for Wearable Applications](https://arxiv.org/pdf/2206.07648.pdf)
* [ECG Heartbeat classification using deep transfer learning with Convolutional Neural Network and STFT technique](https://arxiv.org/pdf/2206.14200.pdf)
* [U2-Net: Going Deeper with Nested U-Structure for Salient Object Detection](https://arxiv.org/abs/2005.09007)
* [UNET 3+: A FULL-SCALE CONNECTED UNET FOR MEDICAL IMAGE SEGMENTATION](https://arxiv.org/pdf/2004.08790.pdf)
* [ResUNet-a: a deep learning framework for semantic segmentation of remotely sensed data](https://arxiv.org/pdf/1904.00592.pdf)
