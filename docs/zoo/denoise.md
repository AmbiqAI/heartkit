# Pre-Trained Denoising Models

## <span class="sk-h2-span">Overview</span>

The following table summarizes the high-level results of the denoising models. The `config` provides the complete configuration JSON file used to train the models. Below we also provide details on the datasets, model architectures, preprocessing, and training procedures used to train the models.

--8<-- "assets/denoise-model-zoo-table.md"

---

## <span class="sk-h2-span">Datasets</span>

The following datasets were used to train the denoising models.

=== "TCN"

    - **[Synthetic via PhysioKit](../datasets/synthetic.md)**
    - **[PTB-XL](../datasets/ptbxl.md)**

---

## <span class="sk-h2-span">Model Architectures</span>

=== "TCN"

    The model is a Temporal Convolutional Network (TCN) that is built using a series of dilated causal convolutional layers. The model is designed to capture long-range temporal dependencies and is well suited for time series data.

---

## <span class="sk-h2-span"> Preprocessing</span>

All models are trained on single channel ECG data. No feature extraction is performed other than applying a bandpass filter to remove noise followed by resampling to target sampling rate. The signal is then normalized by subtracting the mean and dividing by the standard deviation. We also add a small epsilon value to the standard deviation to avoid division by zero.

---

## <span class="sk-h2-span"> Training Procedure </span>

All models are trained using the following setup:

- **[Focal loss function](https://arxiv.org/pdf/1708.02002.pdf)**
- **[Adam optimizer](https://arxiv.org/pdf/1412.6980.pdf)**
- **[Cosine decay learning rate scheduler w/ restarts](https://arxiv.org/pdf/1608.03983.pdf)**
- **Early stopping**

---

## <span class="sk-h2-span">Metrics</span>

=== "TCN"

    TODO: Add metrics

## <span class="sk-h2-span">EVB Performance</span>

The following table provides the latest hardware performance results when running on Apollo4 Plus EVB.

--8<-- "assets/denoise-model-hw-table.md"

---

<!-- ## <span class="sk-h2-span">Comparison</span> -->


<!-- ## <span class="sk-h2-span">Ablation Studies</span> -->
