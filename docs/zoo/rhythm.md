# Pre-Trained Rhythm Models

## <span class="sk-h2-span">Overview</span>

The following table provides the latest pre-trained models for rhythm classification. Below we also provide additional details including training configuration, accuracy metrics, and hardware performance results for the models.


--8<-- "assets/zoo/rhythm/rhythm-model-zoo-table.md"

---

## <span class="sk-h2-span">Model Details</span>

=== "ARR-2-EFF-SM"

    The __ARR-2-EFF-SM__ model is a 2-class arrhythmia classification model that uses EfficientNetV2. The model is trained on raw ECG data and is able to discern normal sinus rhythm (NSR) from atrial fibrillation (AFIB) and atrial flutter (AFL).

    ### Input

    - **Sensor**: ECG
    - **Location**: Wrist
    - **Sampling Rate**: 100 Hz
    - **Frame Size**: 5 seconds

    ### Class Mapping

    Classify both AFIB and AFL as a single class


    | Base Class    | Target Class | Label     |
    | ------------- | ------------ | --------- |
    | 0-Normal      | 0            | NSR       |
    | 7-AFIB, 8-AFL | 1            | AFIB      |


    ### Datasets

    - **[Icentia11k](../datasets/icentia11k.md)**
    - **[PTB-XL](../datasets/ptbxl.md)**
    - **[LSAD](../datasets/lsad.md)**


=== "ARR-4-EFF-SM"

    The __ARR-4-EFF-SM__ model is a 4-class arrhythmia classification model that uses EfficientNetV2. The model is trained on raw ECG data and is able to discern normal sinus rhythm (NSR), sinus bradycardia (SBRAD), atrial fibrillation (AFIB), and general supraventricular tachycardia (GSVT).

    ### Input

    - **Sensor**: ECG
    - **Location**: Wrist
    - **Sampling Rate**: 100 Hz
    - **Frame Size**: 5 seconds

    ### Class Mapping

    Identify rhythm into one of four categories: SR, SBRAD, AFIB, GSVT.

    | Base Class     | Target Class | Label                     |
    | -------------- | ------------ | ------------------------- |
    | 0-SR           | 0            | Sinus Rhythm (SR)         |
    | 1-SBRAD        | 1            | Sinus Bradycardia (SBRAD) |
    | 7-AFIB, 8-AFL  | 2            | AFIB/AFL (AFIB) |
    | 2-STACH, 5-SVT | 3            | General supraventricular tachycardia (GSVT) |

    ### Datasets

    - **[LSAD](../datasets/lsad.md)**

---

## <span class="sk-h2-span">Model Performance</span>

=== "ARR-2-EFF-SM"

    The confusion matrix for the __ARR-2-EFF-SM__ rhythm model is depicted below.

    <div class="sk-plotly-graph-div">
    --8<-- "assets/zoo/rhythm/arr-2-eff-sm-cm.html"
    </div>

=== "ARR-4-EFF-SM"

    The confusion matrix for the __ARR-4-EFF-SM__ rhythm model is depicted below.

    <div class="sk-plotly-graph-div">
    --8<-- "assets/zoo/rhythm/arr-4-eff-sm-cm.html"
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

--8<-- "assets/zoo/rhythm/rhythm-model-hw-table.md"


---

<!--
## <span class="sk-h2-span">Ablation Studies</span>

### Confidence Level

=== "2-Class"

    | Metric   | Baseline | 75% Confidence |
    | -------- | -------- | -------------- |
    | Accuracy | 96.5%    | 99.1%          |
    | F1 Score | 96.4%    | 99.0%          |
    | Drop     |  0.0%    | 12.0%          |

!!! Note "Note"

    The baseline model is simply selecting the argmax of model outputs (e.g. `AFIB/AFL`). A confidence level is used such that a label of inconclusive is assigned when the softmax output is below this threshold. -->
