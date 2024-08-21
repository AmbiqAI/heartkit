# Pre-Trained Beat Models

## <span class="sk-h2-span">Overview</span>

The following table provides the latest pre-trained models for beat classification. Below we also provide additional details including training configuration, accuracy metrics, and hardware performance results for the models.

--8<-- "assets/zoo/beat/beat-model-zoo-table.md"

---

## <span class="sk-h2-span">Model Details</span>

=== "BEAT-2-EFF-SM"

    The __BEAT-2-EFF-SM__ model is a 2-class beat classification model that uses EfficientNetV2. The model is trained on raw ECG data and is able to classify normal sinus rhythm (NSR) and premature atrial/ventricular contractions (PAC/PVC).

    ### Input

    - **Sensor**: ECG
    - **Location**: Wrist
    - **Sampling Rate**: 100 Hz
    - **Frame Size**: 5 seconds

    ### Class Mapping

    Identify PAC and PVC beats.

    | Base Class    | Target Class | Label     |
    | ------------- | ------------ | --------- |
    | 0-NSR         | 0            | NSR       |
    | 1-PAC, 2-PVC  | 1            | PAC|PVC   |

    ### Datasets

    - **[Icentia11k](../datasets/icentia11k.md)**

=== "BEAT-3-EFF-SM"

    The __BEAT-3-EFF-SM__ model is a 2-class beat classification model that uses EfficientNetV2. The model is trained on raw ECG data and is able to classify normal sinus rhythm (NSR), premature/escape atrial contractions, and premature/escape ventricular contractions (PAC/PVC).

    ### Input

    - **Sensor**: ECG
    - **Location**: Wrist
    - **Sampling Rate**: 100 Hz
    - **Frame Size**: 5 seconds

    ### Class Mapping

    Distinguish between normal sinus rhythm (NSR), premature/ectopic atrial contractions (PAC), and premature/ectopic ventricular contractions (PVC).

    | Base Class    | Target Class | Label     |
    | ------------- | ------------ | --------- |
    | 0-NSR         | 0            | NSR       |
    | 1-PAC         | 1            | PAC       |
    | 2-PVC         | 2            | PVC       |

    ### Dataset

    - **[Icentia11k](../datasets/icentia11k.md)**

---

## <span class="sk-h2-span">Model Performance</span>

=== "BEAT-2-EFF-SM"

    The confusion matrix for the __BEAT-2-EFF-SM__ beat model is depicted below.

    <div class="sk-plotly-graph-div">
    --8<-- "assets/zoo/beat/bc-2-eff-sm-cm.html"
    </div>


=== "BEAT-3-EFF-SM"

    The confusion matrix for the __BEAT-3-EFF-SM__ beat model is depicted below.

    <div class="sk-plotly-graph-div">
    --8<-- "assets/zoo/beat/bc-3-eff-sm-cm.html"
    </div>

---

## <span class="sk-h2-span">Downloads</span>

=== "BEAT-2-EFF-SM"

    | Asset                                                                | Description                   |
    | -------------------------------------------------------------------- | ----------------------------- |
    | [configuration.json](https://ambiqai-model-zoo.s3.us-west-2.amazonaws.com/heartkit/beat/beat-2-eff-sm/latest/configuration.json)   | Configuration file            |
    | [model.keras](https://ambiqai-model-zoo.s3.us-west-2.amazonaws.com/heartkit/beat/beat-2-eff-sm/latest/model.keras)            | Keras Model file              |
    | [model.tflite](https://ambiqai-model-zoo.s3.us-west-2.amazonaws.com/heartkit/beat/beat-2-eff-sm/latest/model.tflite)       | TFLite Model file             |
    | [metrics.json](https://ambiqai-model-zoo.s3.us-west-2.amazonaws.com/heartkit/beat/beat-2-eff-sm/latest/beat-2-eff-sm/metrics.json)       | Metrics file                  |

=== "BEAT-3-EFF-SM"

    | Asset                                                                | Description                   |
    | -------------------------------------------------------------------- | ----------------------------- |
    | [configuration.json](https://ambiqai-model-zoo.s3.us-west-2.amazonaws.com/heartkit/beat/beat-3-eff-sm/latest/configuration.json)   | Configuration file            |
    | [model.keras](https://ambiqai-model-zoo.s3.us-west-2.amazonaws.com/heartkit/beat/beat-3-eff-sm/latest/model.keras)            | Keras Model file              |
    | [model.tflite](https://ambiqai-model-zoo.s3.us-west-2.amazonaws.com/heartkit/beat/beat-3-eff-sm/latest/model.tflite)       | TFLite Model file             |
    | [metrics.json](https://ambiqai-model-zoo.s3.us-west-2.amazonaws.com/heartkit/beat/beat-3-eff-sm/latest/beat-3-eff-sm/metrics.json)       | Metrics file                  |

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

--8<-- "assets/zoo/beat/beat-model-hw-table.md"

--- -->
