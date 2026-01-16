# ECG_based Denoising (DEN-TCN-LG)

## Overview

The following table provides the latest pre-trained model for ECG-based denoising. Below we also provide additional details including training configuration, performance metrics, and downloads.

--8<-- "assets/zoo/den-tcn-lg/results.md"

---

## Input

The model is trained on 5-second, raw ECG frames sampled at 100 Hz.

- **Sensor**: ECG
- **Location**: Wrist
- **Sampling Rate**: 100 Hz
- **Frame Size**: 5 seconds

---

## Datasets

The model is trained on the following datasets:

- **[Synthetic](../datasets/synthetic.md)**
- **[PTB-XL](../datasets/ptbxl.md)**

---

## Model Performance

The following table provides the performance metrics for the ECG-based denoising model.

| Metric       | Value |
| ------------ | ----- |
| MAE          | 5.0%  |
| MSE          | 0.8%  |
| COSSIM       | 87.9% |

---

## Downloads


| Asset                                                                | Description                   |
| -------------------------------------------------------------------- | ----------------------------- |
| [configuration.json](https://ambiqai-model-zoo.s3.us-west-2.amazonaws.com/heartkit/denoise/den-tcn-lg/latest/configuration.json)   | Configuration file            |
| [model.keras](https://ambiqai-model-zoo.s3.us-west-2.amazonaws.com/heartkit/denoise/den-tcn-lg/latest/model.keras)            | Keras Model file              |
| [model.tflite](https://ambiqai-model-zoo.s3.us-west-2.amazonaws.com/heartkit/denoise/den-tcn-lg/latest/model.tflite)       | TFLite Model file             |
| [metrics.json](https://ambiqai-model-zoo.s3.us-west-2.amazonaws.com/heartkit/denoise/den-tcn-lg/latest/metrics.json)       | Metrics file                  |

---
