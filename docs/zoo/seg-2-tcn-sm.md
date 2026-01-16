# 2-Stage ECG Segmentation (SEG-2-TCN-SM)

## Overview

The following table provides the latest pre-trained model for 2-stage ECG segmentation. Below we also provide additional details including training configuration, performance metrics, and downloads.

--8<-- "assets/zoo/seg-2-tcn-sm/results.md"

---

## Input

The model is trained on 2.5-second, raw ECG frames sampled at 100 Hz.

- **Sensor**: ECG
- **Location**: Wrist
- **Sampling Rate**: 100 Hz
- **Frame Size**: 2.5 seconds

---

## Class Mapping

The model is able to segment ECG signals into two classes: QRS complexes and none. The class mapping is as follows:

| Base Class    | Target Class | Label     |
| ------------- | ------------ | --------- |
| 0-NONE        | 0            | NONE      |
| 2-QRS         | 1            | QRS       |

---

## Datasets

The model is trained on the following datasets:

- **[Lobachevsky University Electrocardiography dataset (LUDB)](../datasets/ludb.md)**
- **[Synthetic](../datasets/synthetic.md)**

---

## Model Performance

The confusion matrix for the model is depicted below.

<div class="sk-plotly-graph-div">
--8<-- "assets/zoo/seg-2-tcn-sm/confusion_matrix_test.html"
</div>

---

## Downloads

| Asset                                                                | Description                   |
| -------------------------------------------------------------------- | ----------------------------- |
| [configuration.json](https://ambiqai-model-zoo.s3.us-west-2.amazonaws.com/heartkit/segmentation/seg-2-tcn-sm/latest/configuration.json)   | Configuration file            |
| [model.keras](https://ambiqai-model-zoo.s3.us-west-2.amazonaws.com/heartkit/segmentation/seg-2-tcn-sm/latest/model.keras)            | Keras Model file              |
| [model.tflite](https://ambiqai-model-zoo.s3.us-west-2.amazonaws.com/heartkit/segmentation/seg-2-tcn-sm/latest/model.tflite)       | TFLite Model file             |
| [metrics.json](https://ambiqai-model-zoo.s3.us-west-2.amazonaws.com/heartkit/segmentation/seg-2-tcn-sm/latest/metrics.json)       | Metrics file                  |

---
