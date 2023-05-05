# Datasets

Several datasets are readily available online that are suitable for training various heart-related models. The following datasets are ones either used or plan to use. For _arrhythmia_ and _beat classification_, [Icentia11k](#icentia11k-dataset) dataset is used as it contains the largest number of patients in a highly ambulatory setting- users wearing a 1-lead chest band for up to two weeks. For segmentation, synthetic and [LUDB](#ludb-dataset) datasets are being utilized. Please make sure to review each dataset's license for terms and limitations.

---

## Icentia11k Dataset

This dataset consists of ECG recordings from 11,000 patients and 2 billion labelled beats. The data was collected by the CardioSTAT, a single-lead heart monitor device from Icentia. The raw signals were recorded with a 16-bit resolution and sampled at 250 Hz with the CardioSTAT in a modified lead 1 position. We provide derived version of the dataset where each patient is stored in separate [HDF5 files](https://www.hdfgroup.org/solutions/hdf5/) on S3. This makes it faster to download as well as makes it possible to leverage TensorFlow `prefetch` and `interleave` to parallelize data loading.

__Heart Tasks__: Arrhythmia, beat

!!! warning
    The dataset is intended for evaluation purposes only and cannot be used for commercial use without permission. Please visit [Physionet](https://physionet.org/content/icentia11k-continuous-ecg/1.0/) for more details.

---

## LUDB Dataset

The Lobachevsky University Electrocardiography database (LUDB) consists of 200 10-second 12-lead records. The boundaries and peaks of P, T waves and QRS complexes were manually annotated by cardiologists. Each record is annotated with the corresponding diagnosis. Please visit [Physionet](https://physionet.org/content/ludb/1.0.1/) for more details.

__Heart Tasks__: Segmentation, HRV

---

## QT Dataset

Over 100 fifteen-minute two-lead ECG recordings with onset, peak, and end markers for P, QRS, T, and (where present) U waves of from 30 to 50 selected beats in each recording. Please visit [Physionet QTDB](https://physionet.org/content/qtdb/1.0.0/) for more details.

__Heart Tasks__: Segmentation, HRV

---

## MIT BIH Arrhythmia Dataset

This dataset consists of ECG recordings from 47 different subjects recorded at a sampling rate of 360 Hz. 23 records (numbered from 100 to 124 inclusive with some numbers missing) chosen at random from this set, and 25 records (numbered from 200 to 234 inclusive, again with some numbers missing) selected from the same set to include a variety of rare but clinically important phenomena that would not be well-represented by a small random sample of Holter recordings. Each of the 48 records is slightly over 30 minutes long. Please visit [Physionet MITDB](https://physionet.org/content/mitdb/1.0.0/) for more details.

__Heart Tasks__: Arrhythmia

---

## MIT BIH Normal Sinus Rhythm Dataset

This dataset includes 18 long-term ECG recordings of subjects referred to the Arrhythmia Laboratory at Boston's Beth Israel Hospital (now the Beth Israel Deaconess Medical Center). Subjects included in this dataset were found to have had no significant arrhythmias; they include 5 men, aged 26 to 45, and 13 women, aged 20 to 50. Please visit [Physionet NSRDB](https://physionet.org/content/nsrdb/1.0.0/) for more details.

__Heart Tasks__: HRV

---

## PTB Diagnostics Dataset

This dataset consists of ECG records from 290 subjects: 148 diagnosed as MI, 52 healthy controls, and the rest are diagnosed with 7 different disease. Each record contains ECG signals from 12 leads sampled at the frequency of 1000 Hz. Please visit [Physionet PTBDB](https://physionet.org/content/ptbdb/1.0.0/) for more details.
