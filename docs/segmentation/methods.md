# Methods & Materials

## <span class="sk-h2-span">Datasets</span>

For training ECG segmentation models, we leverage Lobachevsky University Electrocardiography dataset (LUDB) and QT dataset (QTDB). In addition, we also utilize PhysioKit's synthetic ECG tool to generate additional training data.

---

## <span class="sk-h2-span">Model Architecture</span>

The pre-trained segmentation models are based on the U-Net architecture [Ronneberger et al., 2015](https://arxiv.org/pdf/1505.04597.pdf). The U-Net architecture is a fully convolutional network that consists of an encoder and decoder. The encoder is a series of convolutional blocks that downsample the input. The decoder is a series of convolutional blocks that upsample the input. Skip connections are used to connect the encoder and decoder. The skip connections allow the decoder to utilize the encoder's feature maps. The U-Net architecture is well suited for segmentation tasks as it allows for the localization of features.

---


## <span class="sk-h2-span">Feature Extraction</span>

The segmentation models are trained directly on single channel ECG data. No feature extraction is performed. However, we do preprocess the data by applying a bandpass filter to remove noise followed by downsampling.

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

Below outlines the class labels used for ECG segmentation.

=== "2-Stage"

    | CLASS    | LABELS                |
    | -------- | --------------------- |
    | 0        | None, P-wave, T-wave  |
    | 1        | QRS                   |

=== "3-Stage"

    | CLASS   | STAGES          |
    | ------- | --------------- |
    | 0       | None            |
    | 1       | QRS             |
    | 2       | P-wave, T-wave  |

=== "4-Stage"

    | CLASS   | STAGES          |
    | ------- | --------------- |
    | 0       | None            |
    | 1       | P-wave          |
    | 2       | QRS             |
    | 3       | T-wave          |

=== "5-Stage"

    | CLASS   | STAGES          |
    | ------- | --------------- |
    | 0       | None            |
    | 1       | P-wave          |
    | 2       | QRS             |
    | 3       | T-wave          |
    | 4       | U-wave          |

---
