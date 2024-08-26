# ARR-4-EFF-SM

## <span class="sk-h2-span">Overview</span>

The following table provides the latest pre-trained models for 4-class arrhythmia classification. Below we also provide additional details including training configuration, accuracy metrics, and hardware performance results for the models.

--8<-- "assets/zoo/rhythm/rhythm-model-zoo-table.md"

---

## <span class="sk-h2-span">Input</span>

- **Sensor**: ECG
- **Location**: Wrist
- **Sampling Rate**: 100 Hz
- **Frame Size**: 5 seconds

---

## <span class="sk-h2-span">Class Mapping</span>

Identify rhythm into one of four categories: SR, SBRAD, AFIB, GSVT.

| Base Class     | Target Class | Label                     |
| -------------- | ------------ | ------------------------- |
| 0-SR           | 0            | Sinus Rhythm (SR)         |
| 1-SBRAD        | 1            | Sinus Bradycardia (SBRAD) |
| 7-AFIB, 8-AFL  | 2            | AFIB/AFL (AFIB) |
| 2-STACH, 5-SVT | 3            | General supraventricular tachycardia (GSVT) |

---

## <span class="sk-h2-span">Datasets</span>

The model is trained on the following datasets:

- **[LSAD](../datasets/lsad.md)**

---

## <span class="sk-h2-span">Model Performance</span>


The confusion matrix for the model is depicted below.

<div class="sk-plotly-graph-div">
--8<-- "assets/zoo/arr-4-eff-sm/confusion_matrix_test.html"
</div>

---

## <span class="sk-h2-span">Downloads</span>


| Asset                                                                | Description                   |
| -------------------------------------------------------------------- | ----------------------------- |
| [configuration.json](https://ambiqai-model-zoo.s3.us-west-2.amazonaws.com/heartkit/rhythm/arr-4-eff-sm/latest/configuration.json)   | Configuration file            |
| [model.keras](https://ambiqai-model-zoo.s3.us-west-2.amazonaws.com/heartkit/rhythm/arr-4-eff-sm/latest/model.keras)            | Keras Model file              |
| [model.tflite](https://ambiqai-model-zoo.s3.us-west-2.amazonaws.com/heartkit/rhythm/arr-4-eff-sm/latest/model.tflite)       | TFLite Model file             |
| [metrics.json](https://ambiqai-model-zoo.s3.us-west-2.amazonaws.com/heartkit/rhythm/arr-4-eff-sm/latest/metrics.json)       | Metrics file                  |

---
