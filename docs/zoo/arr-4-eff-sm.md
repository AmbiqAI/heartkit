# 4-Class Arrhythmia Classification (ARR-4-EFF-SM)

## Overview

The following table provides the latest pre-trained model for 4-class arrhythmia classification. Below we also provide additional details including training configuration, performance metrics, and downloads.

--8<-- "assets/zoo/rhythm/rhythm-model-zoo-table.md"

---

## Input

The model is trained on 5-second, raw ECG frames sampled at 100 Hz.

- **Sensor**: ECG
- **Location**: Wrist
- **Sampling Rate**: 100 Hz
- **Frame Size**: 5 seconds

---

## Class Mapping

The model is trained on raw ECG data and is able to discern normal sinus rhythm (SR) from sinus bradycardia (SBRAD), atrial fibrillation (AFIB), atrial flutter (AFL), supraventricular tachycardia (STACH), and general supraventricular tachycardia (GSVT). The class mapping is as follows:

| Base Class     | Target Class | Label                     |
| -------------- | ------------ | ------------------------- |
| 0-SR           | 0            | Sinus Rhythm (SR)         |
| 1-SBRAD        | 1            | Sinus Bradycardia (SBRAD) |
| 7-AFIB, 8-AFL  | 2            | AFIB/AFL (AFIB) |
| 2-STACH, 5-SVT | 3            | General supraventricular tachycardia (GSVT) |

---

## Datasets

The model is trained on the following datasets:

- **[LSAD](../datasets/lsad.md)**

---

## Model Performance


The confusion matrix for the model is depicted below.

<div class="sk-plotly-graph-div">
--8<-- "assets/zoo/arr-4-eff-sm/confusion_matrix_test.html"
</div>

---

## Downloads


| Asset                                                                | Description                   |
| -------------------------------------------------------------------- | ----------------------------- |
| [configuration.json](https://ambiqai-model-zoo.s3.us-west-2.amazonaws.com/heartkit/rhythm/arr-4-eff-sm/latest/configuration.json)   | Configuration file            |
| [model.keras](https://ambiqai-model-zoo.s3.us-west-2.amazonaws.com/heartkit/rhythm/arr-4-eff-sm/latest/model.keras)            | Keras Model file              |
| [model.tflite](https://ambiqai-model-zoo.s3.us-west-2.amazonaws.com/heartkit/rhythm/arr-4-eff-sm/latest/model.tflite)       | TFLite Model file             |
| [metrics.json](https://ambiqai-model-zoo.s3.us-west-2.amazonaws.com/heartkit/rhythm/arr-4-eff-sm/latest/metrics.json)       | Metrics file                  |

---
