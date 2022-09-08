# ECG Heart Arrhythmia Classification

The objective is to perform real-time Heart Arrhythmia classification using 1-lead ECG and optionally PPG. Classification can be performed on either rhythm (e.g. normal, AFIB, AFL) or beat (e.g. PAC, PVC).

## Datasets

### Icentia 11k Database

This dataset consists of ECG recordings from 11,000 patients and 2 billion labelled beats. The data was collected by the CardioSTAT, a single-lead heart monitor device from Icentia. The raw signals were recorded with a 16-bit resolution and sampled at 250 Hz with the CardioSTAT in a modified lead 1 position.

### MIT-BIH Arrhythmia Database

This dataset consists of ECG recordings from 47 different subjects recorded at a sampling rate of 360 Hz. 23 records (numbered from 100 to 124 inclusive with some numbers missing) chosen at random from this set, and 25 records (numbered from 200 to 234 inclusive, again with some numbers missing) selected from the same set to include a variety of rare but clinically important phenomena that would not be well-represented by a small random sample of Holter recordings. Each of the 48 records is slightly over 30 minutes long.

The subjects were 25 men aged 32 to 89 years, and 22 women aged 23 to 89 years. (Records 201 and 202 came from the same male subject.)

### MIT-BIH Normal Sinus Rhythm Database

This database includes 18 long-term ECG recordings of subjects referred to the Arrhythmia Laboratory at Boston's Beth Israel Hospital (now the Beth Israel Deaconess Medical Center). Subjects included in this database were found to have had no significant arrhythmias; they include 5 men, aged 26 to 45, and 13 women, aged 20 to 50.

### PTB Diagnostics dataset

Dataset consists of ECG records from 290 subjects: 148 diagnosed as MI, 52 healthy control, and the rest are diagnosed with 7 different disease. Each record contains ECG signals from 12 leads sampled at the frequency of 1000 Hz.

## Pipeline

* ECG -> Spectrogram -> 2-D CNN -> Classification [85% accuracy]

* ECG -> Time-series -> 1-D CNN -> Classification

* ECG -> Time-series -> LSTM -> Classification
* ECG -> Scalogram -> 2-D CNN -> Classification

## Reference Papers

* [ECG Heartbeat classification using deep transfer learning with Convolutional Neural Network and STFT technique](https://arxiv.org/abs/2206.14200)
* [Classification of ECG based on Hybrid Features using CNNs for Wearable Applications](https://arxiv.org/pdf/2206.07648.pdf)
* [ECG Heartbeat classification using deep transfer learning with Convolutional Neural Network and STFT technique](https://arxiv.org/pdf/2206.14200.pdf)

### Feature Extraction

1. Spectrogram (STFT)
2. Raw Time-series
3. Scalogram (CWT)

### Model Architecture

1. 1D/2D-CNN
1. LSTM

## Hardware

### ECG / PPG ICs

* [AS7038RB](https://ams.com/en/as7038rb)
* [MAX86150](https://www.maximintegrated.com/en/products/interface/signal-integrity/MAX86150.html)

### ECG / PPG Eval Kits

* [Protocentral MAX86150](https://protocentral.com/product/protocentral-max86150-ppg-and-ecg-breakout-with-qwiic-v2/)
* [MAX86150 Eval Sys](https://www.mouser.com/ProductDetail/Maxim-Integrated/MAX86150EVSYS?qs=chTDxNqvsylQl7eFGURvdw%3D%3D&gclid=Cj0KCQjwlK-WBhDjARIsAO2sErRwZ1A4Pl8LggiOEdglcysa__Eg1f1dHl_KnadjWswq8q6ttpaA72IaAocyEALw_wcB)

## Project Milestones

### 09/05

* EVB: Incorporate ECG sensor (get physical HW working)
* EVB: Finish pipeline/ app state machine

### 09/12

* EVB: Implement pre-processing stage (window filter, normalize)
* EVB: Integrate eRPC for CLI (trigger, data, results)

### 09/19

* EVB: Integrate V1 model (verify stimulus sets produce ~same~ outputs)
* TFL: Quantization and HPO

### 09/26

* Makefile: train, deploy, inference
* CI/CD

### 10/03

* Documentation: Python, C, notebook examples
* Dockerize: install, train, deploy, inference

### 10/10

* Documentation
* Demo UI

## EVB Inference Pipeline

1. We trigger using either button or eRPC command.
2. Data collection from either sensor or data sent over eRPC
3. Model inference is performed
4. Results are displayed w/ display or via eRPC to Python UI
