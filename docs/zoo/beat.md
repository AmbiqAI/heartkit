# Pre-Trained Beat Models

## <span class="sk-h2-span">Overview</span>

The following table summarizes the high-level results of the segmentation models. The `config` provides the complete configuration JSON file used to train the models. Below we also provide details on the datasets, model architectures, preprocessing, and training procedures used to train the models.

--8<-- "assets/beat-model-zoo-table.md"

---

## <span class="sk-h2-span">Datasets</span>

The following datasets were used to train the beat models.

=== "2-Class"

    - **[Icentia11k](../datasets/icentia11k.md)**

=== "3-Class"

    - **[Icentia11k](../datasets/icentia11k.md)**

---

## <span class="sk-h2-span">Model Architectures</span>

All models utilizes a variation of [EfficientNetV2](../models/efficientnet.md) that is adapted for 1-D time series data. The model is a 1-D CNN built using MBConv style blocks that incorporate expansion, inverted residuals, and squeeze and excitation layers. Furthermore, longer filter and stride lengths are utilized in the initial layers to capture more temporal dependencies.

---

## <span class="sk-h2-span"> Preprocessing</span>

The models are trained directly on single channel ECG data. No feature extraction is performed other than applying a band-pass filter to remove noise followed by down-sampling. The signal is then normalized by subtracting the mean and dividing by the standard deviation. We also add a small epsilon value to the standard deviation to avoid division by zero.

---

## <span class="sk-h2-span"> Training Procedure </span>

All models are trained using the following setup:

- **[Focal loss function](https://arxiv.org/pdf/1708.02002.pdf)**
- **[Adam optimizer](https://arxiv.org/pdf/1412.6980.pdf)**
- **[Cosine decay learning rate scheduler w/ restarts](https://arxiv.org/pdf/1608.03983.pdf)**
- **Early stopping**


---


## <span class="sk-h2-span">Class Mapping</span>

Below outlines the class label mappings for the arrhtyhmia models.

=== "2-Class"

    Classify PAC and PVC as a single class.

    | Base Class    | Target Class | Label     |
    | ------------- | ------------ | --------- |
    | 0-NSR         | 0            | NSR       |
    | 1-PAC, 2-PVC  | 1            | PAC|PVC   |


=== "3-Class"

    | Base Class    | Target Class | Label     |
    | ------------- | ------------ | --------- |
    | 0-NSR         | 0            | NSR       |
    | 1-PAC         | 1            | PAC       |
    | 2-PVC         | 2            | PVC       |

---


## <span class="sk-h2-span">Confusion Matrix</span>

=== "2-Class"

    The confusion matrix for the 2-class beat model is depicted below.

    ![2-Class Beat Confusion Matrix](../assets/beat-2-cm.png){ width="480" }

=== "3-Class"

    The confusion matrix for the 3-class model is depicted below.

    ![2-Stage Beat Confusion Matrix](../assets/beat-3-cm.png){ width="480" }

---

## <span class="sk-h2-span">EVB Performance</span>

The following table provides the latest hardware performance results when running on Apollo4 Plus EVB.

--8<-- "assets/beat-model-hw-table.md"

---

## <span class="sk-h2-span">Ablation Studies</span>

### Temporal vs Spatial Channels

The results of three beat models when testing on 1,000 patients (not used during training) are summarized below. The 800x1 model serves as the baseline and classifies individual beats (1 channel) with a fixed time window of 800 ms (160 samples). The 2400x1 model increases the time window to 2,400 ms (480 samples) in order to include surrounding data as context. Increasing the time window increases the accuracy by over `10%` but also causes computation to increase by `3.5x`. The third and best model uses a time window of 800 ms to capture individual beats but includes two additional channels. Using the local average RR interval, the previous and subsequent `beats` are included as side channels. Unlike normal beats, premature and ectopic beats won't be aligned to neighboring beats and serves as useful context. This provides similar temporal resolution as 800x1 but reduces computation by `3.3x` while further improving accuracy by `1.7%`.

| Model      | 800x1  | 2400x1 | 800x3  |
| ---------- | ------ | ------ | ------ |
| Parameters | 73K    | 73K    | 73K    |
| FLOPS      | 2.1M   | 7.6M   | 2.2M   |
| Accuracy   | 78.2%  | 88.6%  | 90.3%  |
| F1 Score   | 77.5%  | 87.2%  | 90.1%  |
