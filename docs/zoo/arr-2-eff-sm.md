# 2-Class Arrhythmia Classification (ARR-2-EFF-SM)


## <span class="sk-h2-span">Overview</span>

The following table provides the latest pre-trained model for 2-class arrhythmia classification. Below we also provide additional details including training configuration, performance metrics, and downloads.

--8<-- "assets/zoo/arr-2-eff-sm/results.md"

---

## <span class="sk-h2-span">Input</span>

- **Sensor**: ECG
- **Location**: Wrist
- **Sampling Rate**: 100 Hz
- **Frame Size**: 5 seconds

## <span class="sk-h2-span">Class Mapping</span>

The model is trained on raw ECG data and is able to discern normal sinus rhythm (NSR) from atrial fibrillation (AFIB) and atrial flutter (AFL). The class mapping is as follows:


| Base Class    | Target Class | Label     |
| ------------- | ------------ | --------- |
| 0-Normal      | 0            | NSR       |
| 7-AFIB, 8-AFL | 1            | AFIB      |


## <span class="sk-h2-span">Datasets</span>

The model is trained on the following datasets:

- **[PTB-XL](../datasets/ptbxl.md)**
- **[LSAD](../datasets/lsad.md)**

---

## <span class="sk-h2-span">Model Performance</span>

The confusion matrix on the test set for the model is depicted below.

<div class="sk-plotly-graph-div">
--8<-- "assets/zoo/arr-2-eff-sm/confusion_matrix_test.html"
</div>

---

## <span class="sk-h2-span">Downloads</span>

| Asset                                                                | Description                   |
| -------------------------------------------------------------------- | ----------------------------- |
| [configuration.json](https://ambiqai-model-zoo.s3.us-west-2.amazonaws.com/heartkit/rhythm/arr-2-eff-sm/latest/configuration.json)   | Configuration file            |
| [model.keras](https://ambiqai-model-zoo.s3.us-west-2.amazonaws.com/heartkit/rhythm/arr-2-eff-sm/latest/model.keras)            | Keras Model file              |
| [model.tflite](https://ambiqai-model-zoo.s3.us-west-2.amazonaws.com/heartkit/rhythm/arr-2-eff-sm/latest/model.tflite)       | TFLite Model file             |
| [metrics.json](https://ambiqai-model-zoo.s3.us-west-2.amazonaws.com/heartkit/rhythm/arr-2-eff-sm/latest/metrics.json)       | Metrics file                  |

---
