# Pre-Trained Beat Models

## <span class="sk-h2-span">Overview</span>

The following table provides the latest pre-trained models for beat classification. Below we also provide additional details including training configuration, accuracy metrics, and hardware performance results for the models.

--8<-- "assets/zoo/beat/beat-model-zoo-table.md"

---

## <span class="sk-h2-span">Model Details</span>

=== "BC-2-EFF-SM"

    The __BC-2-EFF-SM__ model is a 2-class beat classification model that uses EfficientNetV2. The model is trained on raw ECG data and is able to classify normal sinus rhythm (NSR) and premature atrial/ventricular contractions (PAC/PVC).

    - **Location**: Wrist
    - **Classes**: NSR, PAC/PVC
    - **Sampling Rate**: 100 Hz
    - **Frame Size**: 5 seconds
    - **Datasets**: [Icentia11k](../datasets/icentia.md)

    | Base Class    | Target Class | Label     |
    | ------------- | ------------ | --------- |
    | 0-NSR         | 0            | NSR       |
    | 1-PAC, 2-PVC  | 1            | PAC|PVC   |


    - **Training Procedure**:
    - **[Focal loss function](https://arxiv.org/pdf/1708.02002.pdf)**
    - **[Adam optimizer](https://arxiv.org/pdf/1412.6980.pdf)**
    - **[Cosine decay learning rate scheduler w/ restarts](https://arxiv.org/pdf/1608.03983.pdf)**
    - **Early stopping**


=== "BC-3-EFF-SM"

    The __BC-3-EFF-SM__ model is a 2-class beat classification model that uses EfficientNetV2. The model is trained on raw ECG data and is able to classify normal sinus rhythm (NSR), premature/escape atrial contractions, and premature/escape ventricular contractions (PAC/PVC).

    - **Location**: Wrist
    - **Classes**: NSR, PAC, PVC
    - **Sampling Rate**: 100 Hz
    - **Frame Size**: 5 seconds
    - **Datasets**: [Icentia11k](../datasets/icentia.md)

    | Base Class    | Target Class | Label     |
    | ------------- | ------------ | --------- |
    | 0-NSR         | 0            | NSR       |
    | 1-PAC         | 1            | PAC       |
    | 2-PVC         | 2            | PVC       |

---

## <span class="sk-h2-span">Model Performance</span>

=== "BC-2-EFF-SM"

    The confusion matrix for the __BC-2-EFF-SM__ beat model is depicted below.

    <div class="sk-plotly-graph-div">
    --8<-- "assets/zoo/beat/bc-2-eff-sm-cm.html"
    </div>


=== "BC-3-EFF-SM"

    The confusion matrix for the __BC-3-EFF-SM__ beat model is depicted below.

    <div class="sk-plotly-graph-div">
    --8<-- "assets/zoo/beat/bc-3-eff-sm-cm.html"
    </div>

---

## <span class="sk-h2-span">EVB Performance</span>

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

---
