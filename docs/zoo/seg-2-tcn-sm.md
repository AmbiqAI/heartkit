# SEG-2-TCN-SM

## <span class="sk-h2-span">Overview</span>

The following table provides the latest pre-trained models for 2-stage ECG segmentation. Below we also provide additional details including training configuration, accuracy metrics, and hardware performance results for the models.

--8<-- "assets/zoo/seg-2-tcn-sm/results.md"

---

## <span class="sk-h2-span">Model Details</span>

The model is a 2-class ECG segmentation model that uses a Temporal Convolutional Network (TCN) architecture. The model is trained on raw ECG data and is able to delineate QRS complexes.

## <span class="sk-h2-span">Input</span>

- **Sensor**: ECG
- **Location**: Wrist
- **Sampling Rate**: 100 Hz
- **Frame Size**: 2.5 seconds

---

## <span class="sk-h2-span">Class Mapping</span>

Detect only QRS complexes

| Base Class    | Target Class | Label     |
| ------------- | ------------ | --------- |
| 0-NONE        | 0            | NONE      |
| 2-QRS         | 1            | QRS       |

---

## <span class="sk-h2-span">Datasets</span>

The model is trained on the following datasets:

- **[Lobachevsky University Electrocardiography dataset (LUDB)](../datasets/ludb.md)**
- **[Synthetic](../datasets/synthetic.md)**

---

## <span class="sk-h2-span">Model Performance</span>

The confusion matrix for the segmentation model is depicted below.

<div class="sk-plotly-graph-div">
--8<-- "assets/zoo/seg-2-tcn-sm/confusion_matrix_test.html"
</div>

---

## <span class="sk-h2-span">Downloads</span>

| Asset                                                                | Description                   |
| -------------------------------------------------------------------- | ----------------------------- |
| [configuration.json](https://ambiqai-model-zoo.s3.us-west-2.amazonaws.com/heartkit/segmentation/seg-2-tcn-sm/latest/configuration.json)   | Configuration file            |
| [model.keras](https://ambiqai-model-zoo.s3.us-west-2.amazonaws.com/heartkit/segmentation/seg-2-tcn-sm/latest/model.keras)            | Keras Model file              |
| [model.tflite](https://ambiqai-model-zoo.s3.us-west-2.amazonaws.com/heartkit/segmentation/seg-2-tcn-sm/latest/model.tflite)       | TFLite Model file             |
| [metrics.json](https://ambiqai-model-zoo.s3.us-west-2.amazonaws.com/heartkit/segmentation/seg-2-tcn-sm/latest/metrics.json)       | Metrics file                  |

---
