# Pre-Trained Segmentation Models

## <span class="sk-h2-span">Datasets Used</span>

We leverage the following datasets for training the segmentation models:

- **[Icentia11k](../datasets/icentia11k.md)**
- **[Lobachevsky University Electrocardiography dataset (LUDB)](../datasets/ludb.md)**
- **[QT dataset (QTDB)](../datasets/qtdb.md)**
- **[Synthetic via PhysioKit](../datasets/synthetic.md)**

---

## <span class="sk-h2-span">Model Architectures</span>

The models are based on the [U-Net architecture](../models/unet.md). The U-Net architecture is a fully convolutional network that consists of an encoder and decoder useful for image segmentation. The network was adapted for 1D time series data by replacing the 2D convolutions with 1D convolutions.

---

## <span class="sk-h2-span"> Preprocessing</span>

For ECG segmentation, the models are trained directly on single channel ECG data. No feature extraction is performed. However, the data is preprocessed by applying a band-pass filter to remove noise followed by down-sampling. The filtered ECG signals are normalized by subtracting the mean and dividing by the standard deviation. We also add a small epsilon value to the standard deviation to avoid division by zero.

---

## <span class="sk-h2-span"> Training Procedure </span>

For training the models, we utilize the following setup:

- **[Focal loss function](https://arxiv.org/pdf/1708.02002.pdf)**
- **[Adam optimizer](https://arxiv.org/pdf/1412.6980.pdf)**
- **[Cosine decay learning rate scheduler w/ restarts](https://arxiv.org/pdf/1608.03983.pdf)**
- **Early stopping**

---

## <span class="sk-h2-span"> Evaluation Metrics </span>

For each dataset, 10% of the data is held out for testing. From the remaining, 20% of the data is randomly selected for validation. There is no mixing of subjects between the training, validation, and test sets. Furthermore, the test set is held fixed while training and validation are randomly split during training. We evaluate the models performance using a variety of metrics including loss, accuracy, F1 score, average precision (AP).

---

## <span class="sk-h2-span">Model Results</span>

The following results are obtained from the pre-trained segmentation models when testing on 1,000 patients (not used during training).

--8<-- "assets/segmentation-model-hw-table.md"

---

## <span class="sk-h2-span">Confusion Matrix</span>

=== "2-Class"

    ![2-Stage Sleep Stage Confusion Matrix](../assets/segmentation-2-cm.png){ width="480" }

=== "3-Class"

    ![3-Stage Sleep Stage Confusion Matrix](../assets/segmentation-3-cm.png){ width="480" }

=== "4-Class"

    ![4-Stage Sleep Stage Confusion Matrix](../assets/segmentation-4-cm.png){ width="480" }

---

<!-- ## <span class="sk-h2-span">EVB Performance</span> -->
