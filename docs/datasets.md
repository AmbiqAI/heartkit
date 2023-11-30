# Datasets

Several datasets are readily available online that are suitable for training various heart-related models. The following datasets are ones either used or plan to use. Please make sure to review each dataset's license for terms and limitations.

---

## Icentia11k Dataset

### Overview

This dataset consists of ECG recordings from 11,000 patients and 2 billion labelled beats. The data was collected by the CardioSTAT, a single-lead heart monitor device from Icentia. The raw signals were recorded with a 16-bit resolution and sampled at 250 Hz with the CardioSTAT in a modified lead 1 position. We provide derived version of the dataset where each patient is stored in separate [HDF5 files](https://www.hdfgroup.org/solutions/hdf5/) on S3. This makes it faster to download as well as makes it possible to leverage TensorFlow `prefetch` and `interleave` to parallelize data loading.

More info available on [PhysioNet website](https://physionet.org/content/icentia11k-continuous-ecg/1.0)

### Funding

This work is partially funded by a grant from Icentia, Fonds de Recherche en Santé du Québec, and the Institut de valorisation des donnees (IVADO).

### Licensing

The Icentia11k dataset is available for non-commercial use only.

### Tasks

* [Arrhythmia](./arrhythmia/overview.md)
* [Beat](./beat/overview.md)

!!! warning
    The dataset is intended for evaluation purposes only and cannot be used for commercial use without permission. Please visit [Physionet](https://physionet.org/content/icentia11k-continuous-ecg/1.0) for more details.

---

## LUDB Dataset

### Overview

The Lobachevsky University Electrocardiography database (LUDB) consists of 200 10-second 12-lead records. The boundaries and peaks of P, T waves and QRS complexes were manually annotated by cardiologists. Each record is annotated with the corresponding diagnosis.

Please visit [Physionet](https://physionet.org/content/ludb/1.0.1/) for more details.

### Funding

The study was supported by the Ministry of Education of the Russian Federation (contract No. 02.G25.31.0157 of 01.12.2015).

### License

The LUDB is available for commercial use.

### Tasks

* [Segmentation](./segmentation/overview.md)

---

## QT Dataset

### Overview

Over 100 fifteen-minute two-lead ECG recordings with onset, peak, and end markers for P, QRS, T, and (where present) U waves of from 30 to 50 selected beats in each recording.

Please visit [Physionet](https://doi.org/10.13026/C24K53) for more details.

### Funding

The QT Database was created as part of a project funded by the National Library of Medicine.

### License

The QT Database is available for commercial use.

### Tasks

* [Segmentation](./segmentation/overview.md)

---

## MIT BIH Arrhythmia Dataset

### Overview

This dataset consists of ECG recordings from 47 different subjects recorded at a sampling rate of 360 Hz. 23 records (numbered from 100 to 124 inclusive with some numbers missing) chosen at random from this set, and 25 records (numbered from 200 to 234 inclusive, again with some numbers missing) selected from the same set to include a variety of rare but clinically important phenomena that would not be well-represented by a small random sample of Holter recordings. Each of the 48 records is slightly over 30 minutes long.

Please visit [Physionet MITDB](https://doi.org/10.13026/C2F305) for more details.

### Funding

N/A

### License

This database is available for commercial use.

### Tasks

* [Arrhythmia](./arrhythmia/overview.md)

---

## MIT BIH Normal Sinus Rhythm Dataset

### Overview

This dataset includes 18 long-term ECG recordings of subjects referred to the Arrhythmia Laboratory at Boston's Beth Israel Hospital (now the Beth Israel Deaconess Medical Center). Subjects included in this dataset were found to have had no significant arrhythmias; they include 5 men, aged 26 to 45, and 13 women, aged 20 to 50.

Please visit [Physionet](https://doi.org/10.13026/C2NK5R) for more details.

### Funding

N/A

### License

This database is available for commercial use.

### Tasks

* [Arrhythmia](./arrhythmia/overview.md)

---

## PTB Diagnostics Dataset

### Overview

This dataset consists of ECG records from 290 subjects: 148 diagnosed as MI, 52 healthy controls, and the rest are diagnosed with 7 different disease. Each record contains ECG signals from 12 leads sampled at the frequency of 1000 Hz.

Please visit [Physionet](https://doi.org/10.13026/C28C71) for more details.

### Funding

This work was supported by the German Federal Ministry of Education and Research (BMBF) within the framework of the e:Med research and funding concept (grant 01ZX1408A).

### License

This database is available for commercial use.

### Tasks

* [Arrhythmia](./arrhythmia/overview.md)
