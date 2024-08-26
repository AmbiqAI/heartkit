# DEN-TCN-LG

## <span class="sk-h2-span">Overview</span>

The following table provides the latest pre-trained models for ECG-based denoising. Below we also provide additional details including training configuration, accuracy metrics, and hardware performance results for the models.


--8<-- "assets/zoo/den-tcn-lg/results.md"

---

## <span class="sk-h2-span">Input</span>

- **Sensor**: ECG
- **Location**: Wrist
- **Sampling Rate**: 100 Hz
- **Frame Size**: 5 seconds

---

## <span class="sk-h2-span">Datasets</span>

The model is trained on the following datasets:

- **[Synthetic](../datasets/synthetic.md)**
- **[PTB-XL](../datasets/ptbxl.md)**

---

## <span class="sk-h2-span">Model Performance</span>

The following table provides the performance metrics for the ECG-based denoising model.

| Metric       | Value |
| ------------ | ----- |
| MAE          | 5.0%  |
| MSE          | 0.8%  |
| COSSIM       | 87.9% |

---

## <span class="sk-h2-span">Downloads</span>


| Asset                                                                | Description                   |
| -------------------------------------------------------------------- | ----------------------------- |
| [configuration.json](https://ambiqai-model-zoo.s3.us-west-2.amazonaws.com/heartkit/denoise/den-tcn-lg/latest/configuration.json)   | Configuration file            |
| [model.keras](https://ambiqai-model-zoo.s3.us-west-2.amazonaws.com/heartkit/denoise/den-tcn-lg/latest/model.keras)            | Keras Model file              |
| [model.tflite](https://ambiqai-model-zoo.s3.us-west-2.amazonaws.com/heartkit/denoise/den-tcn-lg/latest/model.tflite)       | TFLite Model file             |
| [metrics.json](https://ambiqai-model-zoo.s3.us-west-2.amazonaws.com/heartkit/denoise/den-tcn-lg/latest/metrics.json)       | Metrics file                  |

---
