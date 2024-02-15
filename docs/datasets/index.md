
# :factory: Dataset Factory

HeartKit provides support for a number of datasets to facilitate training the __heart-monitoring tasks__ such as arrhythmia, segmentation, and denoising. Most of the datasets are readily available and can be downloaded and used for training and evaluation. Please make sure to review each dataset's license for terms and limitations.

## <span class="sk-h2-span">Segmentation Datasets</span>

ECG segmentation is the process of identifying the boundaries of the P-wave, QRS complex, and T-wave in an ECG signal. The following datasets are available for segmentation tasks:

* **[LUDB](./ludb.md)**: Lobachevsky University Electrocardiography database consists of 200 10-second 12-lead records. The boundaries and peaks of P, T waves and QRS complexes were manually annotated by cardiologists. Each record is annotated with the corresponding diagnosis.

* **[QTDB](./qtdb.md)**: Over 100 fifteen-minute two-lead ECG recordings with onset, peak, and end markers for P, QRS, T, and (where present) U waves of from 30 to 50 selected beats in each recording.

* **[Synthetic](./synthetic.md)**: A synthetic dataset generated using PhysioKit. The dataset enables the generation of ECG signals with a variety of heart conditions and noise levels.

---

## <span class="sk-h2-span">Arrhythmia Datasets</span>

Arrhythmia detection is the process of identifying abnormal heart rhythms. The following datasets are available for arrhythmia tasks:

* **[Icentia11k](./icentia11k.md)**: This dataset consists of ECG recordings from 11,000 patients and 2 billion labelled beats. The data was collected by the CardioSTAT, a single-lead heart monitor device from Icentia. The raw signals were recorded with a 16-bit resolution and sampled at 250 Hz with the CardioSTAT in a modified lead 1 position.

* **[PTB-XL](./ptbxl.md)**: The PTB-XL is a large publicly available electrocardiography dataset. It contains 21837 clinical 12-lead ECGs from 18885 patients of 10 second length. The ECGs are sampled at 500 Hz and are annotated by up to two cardiologists.

* **[Synthetic](./synthetic.md)**: A synthetic dataset generated using PhysioKit. The dataset enables the generation of ECG signals with a variety of heart conditions and noise levels.

---

## <span class="sk-h2-span">Beat Datasets</span>

Beat classification is the process of identifying abnormal beats in an ECG signal. The following datasets are available for beat classification tasks:

* **[Icentia11k](./icentia11k.md)**: This dataset consists of ECG recordings from 11,000 patients and 2 billion labelled beats. The data was collected by the CardioSTAT, a single-lead heart monitor device from Icentia. The raw signals were recorded with a 16-bit resolution and sampled at 250 Hz with the CardioSTAT in a modified lead 1 position.

* **[PTB-XL](./ptbxl.md)**: The PTB-XL is a large publicly available electrocardiography dataset. It contains 21837 clinical 12-lead ECGs from 18885 patients of 10 second length. The ECGs are sampled at 500 Hz and are annotated by up to two cardiologists.


<!-- ## <span class="sk-h2-span">Sensing Modalities</span>

The two primary sensing modalities to monitor cardiac cycles are electrocardiograph (ECG) and photoplethysmography (PPG). Since HeartKit is targeted for low-power, wearable applications we focus on either single-lead ECG and/or 1-2 channels of PPG that can be easily captured on wrist, ear, or chest. The following table provides a comparison of the two modalities:

| Modality | ECG | PPG |
| -------- | --- | --- |
| Description | Electrical activity of heart | Blood volume changes in tissue |
| Location | Chest, Wrist, Ear | Wrist, Ear |
| Channels | 1-3 | 1-2 |
| SNR | High | Low |
| Noise | Low | High |
| Fidelity | High | Low |
| Power | High | Low |
| Contact | Yes | No |

--- -->
