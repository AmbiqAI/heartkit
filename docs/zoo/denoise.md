# Pre-Trained Denoising Models

## <span class="sk-h2-span">Overview</span>

The following table provides the latest pre-trained models for signal denoising. Below we also provide additional details including training configuration, accuracy metrics, and hardware performance results for the models.


--8<-- "assets/zoo/denoise/denoise-model-zoo-table.md"

---

## <span class="sk-h2-span">Model Details</span>

=== "DEN-TCN-SM"

    The __DEN-TCN-SM__ model is a small denoising model that uses a Temporal Convolutional Network (TCN). The model is trained on raw ECG data and is able to remove noise from the signal.

    ### Input

    - **Sensor**: ECG
    - **Location**: Wrist
    - **Sampling Rate**: 100 Hz
    - **Frame Size**: 5 seconds

    ### Datasets

    - **[Synthetic](../datasets/synthetic.md)**
    - **[PTB-XL](../datasets/ptbxl.md)**

=== "DEN-TCN-LG"

    The __DEN-TCN-SM__ model is a larger denoising model that uses a Temporal Convolutional Network (TCN). The model is trained on raw ECG data and is able to remove noise from the signal.

    ### Input

    - **Sensor**: ECG
    - **Location**: Wrist
    - **Sampling Rate**: 100 Hz
    - **Frame Size**: 5 seconds

    ### Datasets

    - **[Synthetic](../datasets/synthetic.md)**
    - **[PTB-XL](../datasets/ptbxl.md)**

=== "DEN-PPG-TCN-SM"

    The __DEN-PPG-TCN-SM__ model is a small denoising model that uses a Temporal Convolutional Network (TCN). The model is trained on raw PPG data and is able to remove noise from the signal.

    ### Input

    - **Sensor**: PPG
    - **Location**: Wrist
    - **Sampling Rate**: 100 Hz
    - **Frame Size**: 5 seconds

    ### Datasets
    - **[Synthetic](../datasets/synthetic.md)**

---

## <span class="sk-h2-span">Model Performance</span>

=== "DEN-TCN-SM"

    The following table provides the performance metrics for the __DEN-TCN-SM__ denoising model.

    | Metric       | Value |
    | ------------ | ----- |
    | MAE          | 6.6%  |
    | MSE          | 1.1%  |
    | COSSIM       | 85.9% |

=== "DEN-TCN-LG"

    The following table provides the performance metrics for the __DEN-TCN-LG__ denoising model.

    | Metric       | Value |
    | ------------ | ----- |
    | MAE          | 5.0% |
    | MSE          | 0.8%  |
    | COSSIM       | 87.9% |

=== "DEN-PPG-TCN-SM"

    The following table provides the performance metrics for the __DEN-PPG-TCN-SM__ denoising model.

    | Metric       | Value |
    | ------------ | ----- |
    | MAE          | 11.0% |
    | MSE          | 2.2%  |
    | COSSIM       | 92.1% |

---

## <span class="sk-h2-span">Downloads</span>

=== "DEN-TCN-SM"

    | Asset                                                                | Description                   |
    | -------------------------------------------------------------------- | ----------------------------- |
    | [configuration.json](https://ambiqai-model-zoo.s3.us-west-2.amazonaws.com/heartkit/denoise/denoise-tcn-sm/latest/configuration.json)   | Configuration file            |
    | [model.keras](https://ambiqai-model-zoo.s3.us-west-2.amazonaws.com/heartkit/denoise/denoise-tcn-sm/latest/model.keras)            | Keras Model file              |
    | [model.tflite](https://ambiqai-model-zoo.s3.us-west-2.amazonaws.com/heartkit/denoise/denoise-tcn-sm/latest/model.tflite)       | TFLite Model file             |
    | [metrics.json](https://ambiqai-model-zoo.s3.us-west-2.amazonaws.com/heartkit/denoise/denoise-tcn-sm/latest/denoise-tcn-sm/metrics.json)       | Metrics file                  |

=== "DEN-TCN-LG"

    | Asset                                                                | Description                   |
    | -------------------------------------------------------------------- | ----------------------------- |
    | [configuration.json](https://ambiqai-model-zoo.s3.us-west-2.amazonaws.com/heartkit/denoise/denoise-tcn-lg/latest/configuration.json)   | Configuration file            |
    | [model.keras](https://ambiqai-model-zoo.s3.us-west-2.amazonaws.com/heartkit/denoise/denoise-tcn-lg/latest/model.keras)            | Keras Model file              |
    | [model.tflite](https://ambiqai-model-zoo.s3.us-west-2.amazonaws.com/heartkit/denoise/denoise-tcn-lg/latest/model.tflite)       | TFLite Model file             |
    | [metrics.json](https://ambiqai-model-zoo.s3.us-west-2.amazonaws.com/heartkit/denoise/denoise-tcn-lg/latest/denoise-tcn-lg/metrics.json)       | Metrics file                  |

=== "DEN-PPG-TCN-SM"

    | Asset                                                                | Description                   |
    | -------------------------------------------------------------------- | ----------------------------- |
    | [configuration.json](https://ambiqai-model-zoo.s3.us-west-2.amazonaws.com/heartkit/denoise/denoise-ppg-tcn-sm/latest/configuration.json)   | Configuration file            |
    | [model.keras](https://ambiqai-model-zoo.s3.us-west-2.amazonaws.com/heartkit/denoise/denoise-ppg-tcn-sm/latest/model.keras)            | Keras Model file              |
    | [model.tflite](https://ambiqai-model-zoo.s3.us-west-2.amazonaws.com/heartkit/denoise/denoise-ppg-tcn-sm/latest/model.tflite)       | TFLite Model file             |
    | [metrics.json](https://ambiqai-model-zoo.s3.us-west-2.amazonaws.com/heartkit/denoise/denoise-ppg-tcn-sm/latest/denoise-ppg-tcn-sm/metrics.json)       | Metrics file                  |


<!-- ## <span class="sk-h2-span">EVB Performance</span>

The following table provides the latest hardware performance results when running on Apollo4 Plus EVB.

--8<-- "assets/zoo/denoise/denoise-model-hw-table.md"

--- -->

<!-- ## <span class="sk-h2-span">EVB Performance</span>

The following table provides the latest performance and accuracy results of all models when running on Apollo4 Plus EVB. These results are obtained using neuralSPOTs [Autodeploy tool](https://ambiqai.github.io/neuralSPOT/docs/From%20TF%20to%20EVB%20-%20testing%2C%20profiling%2C%20and%20deploying%20AI%20models.html). From neuralSPOT repo, the following command can be used to capture EVB results via Autodeploy:

``` console
python -m ns_autodeploy \
--tflite-filename model.tflite \
--model-name model \
--cpu-mode 192 \
--arena-size-scratch-buffer-padding 0 \
--max-arena-size 80 \

```

--8<-- "assets/zoo/denoise/denoise-model-hw-table.md" -->
