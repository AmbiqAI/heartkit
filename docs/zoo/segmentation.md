# Pre-Trained Segmentation Models

## <span class="sk-h2-span">Overview</span>

The following table provides the latest pre-trained models for ECG segmentation. Below we also provide additional details including training configuration, accuracy metrics, and hardware performance results for the models.

--8<-- "assets/zoo/segmentation/segmentation-model-zoo-table.md"

---

## <span class="sk-h2-span">Model Details</span>

=== "SEG-2-TCN-SM"

    The __SEG-2-TCN-SM__ model is a 2-class ECG segmentation model that uses a Temporal Convolutional Network (TCN) architecture. The model is trained on raw ECG data and is able to delineate QRS complexes.

    ### Input

    - **Sensor**: ECG
    - **Location**: Wrist
    - **Sampling Rate**: 100 Hz
    - **Frame Size**: 2.5 seconds

    ### Class Mapping

    Detect only QRS complexes

    | Base Class    | Target Class | Label     |
    | ------------- | ------------ | --------- |
    | 0-NONE        | 0            | NONE      |
    | 2-QRS         | 1            | QRS       |

    ### Datasets

    - **[Lobachevsky University Electrocardiography dataset (LUDB)](../datasets/ludb.md)**
    - **[Synthetic](../datasets/synthetic.md)**

=== "SEG-4-TCN-SM"

    The __SEG-4-TCN-SM__ model is a 4-class ECG segmentation model that uses a Temporal Convolutional Network (TCN) architecture. The model is trained on raw ECG data and is able to delineate P-wave, QRS complexes, and T-wave.

    ### Input

    - **Sensor**: ECG
    - **Location**: Wrist
    - **Sampling Rate**: 100 Hz
    - **Frame Size**: 2.5 seconds

    ### Class Mapping

    Identify each of the P-wave, QRS complex, and T-wave.

    | Base Class       | Target Class | Label        |
    | ---------------- | ------------ | ------------ |
    | 0-NONE           | 0            | NONE         |
    | 1-PWAVE          | 1            | PWAVE        |
    | 2-QRS            | 2            | QRS          |
    | 3-TWAVE          | 3            | TWAVE        |

    ### Datasets

    - **[Lobachevsky University Electrocardiography dataset (LUDB)](../datasets/ludb.md)**
    - **[Synthetic](../datasets/synthetic.md)**

=== "SEG-4-TCN-LG"

    The __SEG-4-TCN-LG__ model is a 4-class ECG segmentation model that uses a Temporal Convolutional Network (TCN) architecture. The model is trained on raw ECG data and is able to delineate P-wave, QRS complexes, and T-wave.

    ### Input

    - **Sensor**: ECG
    - **Location**: Wrist
    - **Sampling Rate**: 100 Hz
    - **Frame Size**: 2.5 seconds

    ### Class Mapping

    Identify each of the P-wave, QRS complex, and T-wave.

    | Base Class       | Target Class | Label        |
    | ---------------- | ------------ | ------------ |
    | 0-NONE           | 0            | NONE         |
    | 1-PWAVE          | 1            | PWAVE        |
    | 2-QRS            | 2            | QRS          |
    | 3-TWAVE          | 3            | TWAVE        |

    ### Datasets

    - **[Lobachevsky University Electrocardiography dataset (LUDB)](../datasets/ludb.md)**
    - **[Synthetic](../datasets/synthetic.md)**

---


## <span class="sk-h2-span">Model Performance</span>

=== "SEG-2-TCN-SM"

    The confusion matrix for the __SEG-2-TCN-SM__ segmentation model is depicted below.

    <div class="sk-plotly-graph-div">
    --8<-- "assets/zoo/segmentation/seg-2-tcn-sm-cm.html"
    </div>

=== "SEG-4-TCN-SM"

    The confusion matrix for the __SEG-4-TCN-SM__ segmentation model is depicted below.

    <div class="sk-plotly-graph-div">
    --8<-- "assets/zoo/segmentation/seg-4-tcn-sm-cm.html"
    </div>

=== "SEG-4-TCN-LG"

    The confusion matrix for the __SEG-4-TCN-LG__ segmentation model is depicted below.

    <div class="sk-plotly-graph-div">
    --8<-- "assets/zoo/segmentation/seg-4-tcn-lg-cm.html"
    </div>

---

## <span class="sk-h2-span">Downloads</span>

=== "SEG-2-TCN-SM"

    | Asset                                                                | Description                   |
    | -------------------------------------------------------------------- | ----------------------------- |
    | [configuration.json](https://ambiqai-model-zoo.s3.us-west-2.amazonaws.com/heartkit/segmentation/seg-2-tcn-sm/latest/configuration.json)   | Configuration file            |
    | [model.keras](https://ambiqai-model-zoo.s3.us-west-2.amazonaws.com/heartkit/segmentation/seg-2-tcn-sm/latest/model.keras)            | Keras Model file              |
    | [model.tflite](https://ambiqai-model-zoo.s3.us-west-2.amazonaws.com/heartkit/segmentation/seg-2-tcn-sm/latest/model.tflite)       | TFLite Model file             |
    | [metrics.json](https://ambiqai-model-zoo.s3.us-west-2.amazonaws.com/heartkit/segmentation/seg-2-tcn-sm/latest/seg-2-tcn-sm/metrics.json)       | Metrics file                  |


=== "SEG-4-TCN-SM"

    | Asset                                                                | Description                   |
    | -------------------------------------------------------------------- | ----------------------------- |
    | [configuration.json](https://ambiqai-model-zoo.s3.us-west-2.amazonaws.com/heartkit/segmentation/seg-4-tcn-sm/latest/configuration.json)   | Configuration file            |
    | [model.keras](https://ambiqai-model-zoo.s3.us-west-2.amazonaws.com/heartkit/segmentation/seg-4-tcn-sm/latest/model.keras)            | Keras Model file              |
    | [model.tflite](https://ambiqai-model-zoo.s3.us-west-2.amazonaws.com/heartkit/segmentation/seg-4-tcn-sm/latest/model.tflite)       | TFLite Model file             |
    | [metrics.json](https://ambiqai-model-zoo.s3.us-west-2.amazonaws.com/heartkit/segmentation/seg-4-tcn-sm/latest/seg-4-tcn-sm/metrics.json)       | Metrics file                  |

=== "SEG-4-TCN-LG"

    | Asset                                                                | Description                   |
    | -------------------------------------------------------------------- | ----------------------------- |
    | [configuration.json](https://ambiqai-model-zoo.s3.us-west-2.amazonaws.com/heartkit/segmentation/seg-4-tcn-lg/latest/configuration.json)   | Configuration file            |
    | [model.keras](https://ambiqai-model-zoo.s3.us-west-2.amazonaws.com/heartkit/segmentation/seg-4-tcn-lg/latest/model.keras)            | Keras Model file              |
    | [model.tflite](https://ambiqai-model-zoo.s3.us-west-2.amazonaws.com/heartkit/segmentation/seg-4-tcn-lg/latest/model.tflite)       | TFLite Model file             |
    | [metrics.json](https://ambiqai-model-zoo.s3.us-west-2.amazonaws.com/heartkit/segmentation/seg-4-tcn-lg/latest/seg-4-tcn-lg/metrics.json)       | Metrics file                  |



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

--8<-- "assets/zoo/segmentation/segmentation-model-hw-table.md" -->
