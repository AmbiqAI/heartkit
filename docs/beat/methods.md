# Methods & Materials

## <span class="sk-h2-span">Datasets</span>

For training beat classification models, we use the [Icentia11k dataset](https://physionet.org/content/icentia11k-continuous-ecg/1.0.0/). This dataset consists of single lead ECG recordings from 11,000 patients and 2 billion labelled beats.

---

## <span class="sk-h2-span">Model Architecture</span>

The beat model utilizes a 1-D CNN built using MBConv style blocks that incorporate expansion, inverted residuals, and squeeze and excitation layers. Furthermore, longer filter and stide lengths are utilized in the initial layers to capture more temporal dependencies.

---


## <span class="sk-h2-span">Feature Extraction</span>

The beat classification models is trained directly on single channel ECG data. No feature extraction is performed. However, we do preprocess the data by applying a bandpass filter to remove noise followed by downsampling.

---

## <span class="sk-h2-span">Feature Normalization</span>

The filtered ECG signals are normalized by subtracting the mean and dividing by the standard deviation. We also add a small epsilon value to the standard deviation to avoid division by zero.

---

## <span class="sk-h2-span">Training Procedure</span>

For training the models, we utilize the following setup. We utilize a focal loss function [Lin et al., 2018](https://arxiv.org/pdf/1708.02002.pdf), Adam optimizer [Kingma et al., 2014](https://arxiv.org/pdf/1412.6980.pdf), cosine decay learning rate scheduler with restarts [Loshchilov et al., 2016](https://arxiv.org/pdf/1608.03983.pdf), and early stopping based on loss metric. The focal loss function along with oversampling is used to deal with the significant class imbalance.

---

## <span class="sk-h2-span">Evaluation Metrics</span>

For each dataset, 10% of the data is held out for testing. From the remaining, 20% of the data is randomly selected for validation. There is no mixing of subjects between the training, validation, and test sets. Furthermore, the test set is held fixed while training and validation are randomly split during training. We evaluate the models performance using a variety of metrics including loss, accuracy, F1 score, average precision (AP).

---

## <span class="sk-h2-span">Classes</span>

Below outlines the class labels used for beat classification.

=== "2-Stage"

    | CLASS    | LABELS           |
    | -------- | ---------------- |
    | 0        | NSR              |
    | 1        | PAC/PVC          |

=== "3-Stage"

    | CLASS   | LABELS           |
    | ------- | ---------------- |
    | 0       | NSR              |
    | 1       | PAC              |
    | 2       | PVC              |

---
