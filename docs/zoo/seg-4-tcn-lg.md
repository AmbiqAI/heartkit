# 4-Stage ECG Segmentation (SEG-4-TCN-LG)

## <span class="sk-h2-span">Overview</span>

The following table provides the latest pre-trained model for 4-class ECG segmentation. Below we also provide additional details including training configuration, performance metrics, and downloads.

--8<-- "assets/zoo/seg-4-tcn-lg/results.md"

---

### <span class="sk-h2-span">Input</span>

The model is trained on 2.5-second, raw ECG frames sampled at 100 Hz.

- **Sensor**: ECG
- **Location**: Wrist
- **Sampling Rate**: 100 Hz
- **Frame Size**: 2.5 seconds

---

### <span class="sk-h2-span">Class Mapping</span>

The model is able to segment ECG signals into four classes: P-wave, QRS complex, T-wave, and none. The class mapping is as follows:

| Base Class       | Target Class | Label        |
| ---------------- | ------------ | ------------ |
| 0-NONE           | 0            | NONE         |
| 1-PWAVE          | 1            | PWAVE        |
| 2-QRS            | 2            | QRS          |
| 3-TWAVE          | 3            | TWAVE        |

---

### <span class="sk-h2-span">Datasets</span>

The model is trained on the following datasets:

- **[Lobachevsky University Electrocardiography dataset (LUDB)](../datasets/ludb.md)**
- **[Synthetic](../datasets/synthetic.md)**

---

## <span class="sk-h2-span">Model Performance</span>

The confusion matrix for the segmentation model is depicted below.

<div class="sk-plotly-graph-div">
--8<-- "assets/zoo/seg-4-tcn-lg/confusion_matrix_test.html"
</div>

---

## <span class="sk-h2-span">Downloads</span>

| Asset                                                                | Description                   |
| -------------------------------------------------------------------- | ----------------------------- |
| [configuration.json](https://ambiqai-model-zoo.s3.us-west-2.amazonaws.com/heartkit/segmentation/seg-4-tcn-lg/latest/configuration.json)   | Configuration file            |
| [model.keras](https://ambiqai-model-zoo.s3.us-west-2.amazonaws.com/heartkit/segmentation/seg-4-tcn-lg/latest/model.keras)            | Keras Model file              |
| [model.tflite](https://ambiqai-model-zoo.s3.us-west-2.amazonaws.com/heartkit/segmentation/seg-4-tcn-lg/latest/model.tflite)       | TFLite Model file             |
| [metrics.json](https://ambiqai-model-zoo.s3.us-west-2.amazonaws.com/heartkit/segmentation/seg-4-tcn-lg/latest/metrics.json)       | Metrics file                  |

---
