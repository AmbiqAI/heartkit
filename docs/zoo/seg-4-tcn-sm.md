# 4-Stage ECG Segmentation (SEG-4-TCN-SM)

## Overview

The following table provides the latest pre-trained model for 4-class ECG segmentation. Below we also provide additional details including training configuration, performance metrics, and downloads.

--8<-- "assets/zoo/seg-4-tcn-sm/results.md"

---

## Input

- **Sensor**: ECG
- **Location**: Wrist
- **Sampling Rate**: 100 Hz
- **Frame Size**: 2.5 seconds

---

## Class Mapping

Identify each of the P-wave, QRS complex, and T-wave.

| Base Class       | Target Class | Label        |
| ---------------- | ------------ | ------------ |
| 0-NONE           | 0            | NONE         |
| 1-PWAVE          | 1            | PWAVE        |
| 2-QRS            | 2            | QRS          |
| 3-TWAVE          | 3            | TWAVE        |

---

## Datasets

The model is trained on the following datasets:

- **[Lobachevsky University Electrocardiography dataset (LUDB)](../datasets/ludb.md)**
- **[Synthetic](../datasets/synthetic.md)**

---

## Model Performance

The confusion matrix for the segmentation model is depicted below.

<div class="sk-plotly-graph-div">
--8<-- "assets/zoo/seg-4-tcn-sm/confusion_matrix_test.html"
</div>

---

## Downloads

| Asset                                                                | Description                   |
| -------------------------------------------------------------------- | ----------------------------- |
| [configuration.json](https://ambiqai-model-zoo.s3.us-west-2.amazonaws.com/heartkit/segmentation/seg-4-tcn-sm/latest/configuration.json)   | Configuration file            |
| [model.keras](https://ambiqai-model-zoo.s3.us-west-2.amazonaws.com/heartkit/segmentation/seg-4-tcn-sm/latest/model.keras)            | Keras Model file              |
| [model.tflite](https://ambiqai-model-zoo.s3.us-west-2.amazonaws.com/heartkit/segmentation/seg-4-tcn-sm/latest/model.tflite)       | TFLite Model file             |
| [metrics.json](https://ambiqai-model-zoo.s3.us-west-2.amazonaws.com/heartkit/segmentation/seg-4-tcn-sm/latest/metrics.json)       | Metrics file                  |

---
