# 2-Class Beat Classification (BEAT-2-EFF-SM)

## <span class="sk-h2-span">Overview</span>

The following table provides the latest pre-trained model for 2-class beat classification task. Below we also provide additional details including training configuration, performance metrics, and downloads.

--8<-- "assets/zoo/beat-2-eff-sm/results.md"

---

## <span class="sk-h2-span">Input</span>

The model is trained on 5-second, raw ECG frames sampled at 100 Hz.

- **Sensor**: ECG
- **Location**: Wrist
- **Sampling Rate**: 100 Hz
- **Frame Size**: 5 seconds

---

## <span class="sk-h2-span">Class Mapping</span>

The model is able to classify normal sinus rhythm (NSR) and premature atrial/ventricular contractions (PAC/PVC).
The class mapping is as follows:

| Base Class    | Target Class | Label     |
| ------------- | ------------ | --------- |
| 0-NSR         | 0            | NSR       |
| 1-PAC, 2-PVC  | 1            | PAC|PVC   |

---

## <span class="sk-h2-span">Datasets</span>

The model is trained on the following datasets:

- **[Icentia11k](../datasets/icentia11k.md)**

---

## <span class="sk-h2-span">Model Performance</span>

The confusion matrix for model is depicted below.

<div class="sk-plotly-graph-div">
--8<-- "assets/zoo/beat-2-eff-sm/confusion_matrix_test.html"
</div>


## <span class="sk-h2-span">Downloads</span>

| Asset                                                                | Description                   |
| -------------------------------------------------------------------- | ----------------------------- |
| [configuration.json](https://ambiqai-model-zoo.s3.us-west-2.amazonaws.com/heartkit/beat/beat-2-eff-sm/latest/configuration.json)   | Configuration file            |
| [model.keras](https://ambiqai-model-zoo.s3.us-west-2.amazonaws.com/heartkit/beat/beat-2-eff-sm/latest/model.keras)            | Keras Model file              |
| [model.tflite](https://ambiqai-model-zoo.s3.us-west-2.amazonaws.com/heartkit/beat/beat-2-eff-sm/latest/model.tflite)       | TFLite Model file             |
| [metrics.json](https://ambiqai-model-zoo.s3.us-west-2.amazonaws.com/heartkit/beat/beat-2-eff-sm/latest/metrics.json)       | Metrics file                  |

---
