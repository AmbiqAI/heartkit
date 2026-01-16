# 2-Stage PPG Segmentation (SEG-PPG-2-TCN-SM)

## Overview

The following table provides the latest pre-trained model for 2-stage PPG segmentation. Below we also provide additional details including training configuration, performance metrics, and downloads.

--8<-- "assets/zoo/seg-ppg-2-tcn-sm/results.md"

---

## Input

The model is trained on 2.5-second, raw ECG frames sampled at 100 Hz.

- **Sensor**: PPG
- **Location**: Wrist
- **Sampling Rate**: 100 Hz
- **Frame Size**: 2.5 seconds

---

## Class Mapping

The model is able to segment PPG Signals into systolic and diastolic phases. The class mapping is as follows:

| Base Class    | Target Class | Label     |
| ------------- | ------------ | --------- |
| 6-SYSTOLIC    | 0            | SYS       |
| 7-DIASTOLIC   | 1            | DIA       |

---

## Datasets

The model is trained on the following datasets:

- **[PPG Synthetic](../datasets/synthetic.md)**

---

## Model Performance

The confusion matrix for the model is depicted below.

<div class="sk-plotly-graph-div">
--8<-- "assets/zoo/seg-ppg-2-tcn-sm/confusion_matrix_test.html"
</div>

---

## Downloads

| Asset                                                                | Description                   |
| -------------------------------------------------------------------- | ----------------------------- |
| [configuration.json](https://ambiqai-model-zoo.s3.us-west-2.amazonaws.com/heartkit/segmentation/seg-ppg-2-tcn-sm/latest/configuration.json)   | Configuration file            |
| [model.keras](https://ambiqai-model-zoo.s3.us-west-2.amazonaws.com/heartkit/segmentation/seg-ppg-2-tcn-sm/latest/model.keras)            | Keras Model file              |
| [model.tflite](https://ambiqai-model-zoo.s3.us-west-2.amazonaws.com/heartkit/segmentation/seg-ppg-2-tcn-sm/latest/model.tflite)       | TFLite Model file             |
| [metrics.json](https://ambiqai-model-zoo.s3.us-west-2.amazonaws.com/heartkit/segmentation/seg-ppg-2-tcn-sm/latest/metrics.json)       | Metrics file                  |

---
