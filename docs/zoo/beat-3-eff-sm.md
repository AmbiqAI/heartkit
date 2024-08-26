# BEAT-3-EFF-SM

## <span class="sk-h2-span">Overview</span>

The following table provides the latest pre-trained models for 3-class beat classification. Below we also provide additional details including training configuration, accuracy metrics, and hardware performance results for the models.

--8<-- "assets/zoo/beat-3-eff-sm/results.md"

---

## <span class="sk-h2-span">Input</span>

- **Sensor**: ECG
- **Location**: Wrist
- **Sampling Rate**: 100 Hz
- **Frame Size**: 5 seconds

---

## <span class="sk-h2-span">Class Mapping</span>

Distinguish between normal sinus rhythm (NSR), premature/ectopic atrial contractions (PAC), and premature/ectopic ventricular contractions (PVC).

| Base Class    | Target Class | Label     |
| ------------- | ------------ | --------- |
| 0-NSR         | 0            | NSR       |
| 1-PAC         | 1            | PAC       |
| 2-PVC         | 2            | PVC       |

---

## <span class="sk-h2-span">Dataset</span>

The model is trained on the following datasets:

- **[Icentia11k](../datasets/icentia11k.md)**

---

## <span class="sk-h2-span">Model Performance</span>

The confusion matrix for the model is depicted below.

<div class="sk-plotly-graph-div">
--8<-- "assets/zoo/beat-3-eff-sm/confusion_matrix_test.html"

</div>

---

## <span class="sk-h2-span">Downloads</span>

| Asset                                                                | Description                   |
| -------------------------------------------------------------------- | ----------------------------- |
| [configuration.json](https://ambiqai-model-zoo.s3.us-west-2.amazonaws.com/heartkit/beat/beat-3-eff-sm/latest/configuration.json)   | Configuration file            |
| [model.keras](https://ambiqai-model-zoo.s3.us-west-2.amazonaws.com/heartkit/beat/beat-3-eff-sm/latest/model.keras)            | Keras Model file              |
| [model.tflite](https://ambiqai-model-zoo.s3.us-west-2.amazonaws.com/heartkit/beat/beat-3-eff-sm/latest/model.tflite)       | TFLite Model file             |
| [metrics.json](https://ambiqai-model-zoo.s3.us-west-2.amazonaws.com/heartkit/beat/beat-3-eff-sm/latest/metrics.json)       | Metrics file                  |

---
