# PPG-Based Denoising (DEN-PPG-TCN-SM)

## <span class="sk-h2-span">Overview</span>

The following table provides the latest pre-trained model for PPG-based denoising. Below we also provide additional details including training configuration, performance metrics, and downloads.

--8<-- "assets/zoo/den-ppg-tcn-sm/results.md"

---

## <span class="sk-h2-span">Input</span>

The model is trained on 5-second, raw PPG frames sampled at 100 Hz.

- **Sensor**: PPG
- **Location**: Wrist
- **Sampling Rate**: 100 Hz
- **Frame Size**: 5 seconds

---

## <span class="sk-h2-span">Datasets</span>

The model is trained on synthetic PPG data.

- **[Synthetic](../datasets/synthetic.md)**

---

## <span class="sk-h2-span">Model Performance</span>

The following table provides the performance metrics for the PPG denoising model.

| Metric       | Value |
| ------------ | ----- |
| MAE          | 11.0% |
| MSE          | 2.2%  |
| COSSIM       | 92.1% |

---

## <span class="sk-h2-span">Downloads</span>


| Asset                                                                | Description                   |
| -------------------------------------------------------------------- | ----------------------------- |
| [configuration.json](https://ambiqai-model-zoo.s3.us-west-2.amazonaws.com/heartkit/denoise/den-ppg-tcn-sm/latest/configuration.json)   | Configuration file            |
| [model.keras](https://ambiqai-model-zoo.s3.us-west-2.amazonaws.com/heartkit/denoise/den-ppg-tcn-sm/latest/model.keras)            | Keras Model file              |
| [model.tflite](https://ambiqai-model-zoo.s3.us-west-2.amazonaws.com/heartkit/denoise/den-ppg-tcn-sm/latest/model.tflite)       | TFLite Model file             |
| [metrics.json](https://ambiqai-model-zoo.s3.us-west-2.amazonaws.com/heartkit/denoise/den-ppg-tcn-sm/latest/metrics.json)       | Metrics file                  |

---
