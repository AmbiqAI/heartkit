# Heart Arrhythmia Classification

The objective is to perform real-time heart arrhythmia classification using 1-lead ECG and optionally PPG. Classification can be performed on either rhythm (e.g. normal, AFIB, AFL) or beat (e.g. PAC, PVC). The current model is able to perform AFIB arrhythmia classification. In the near future, this will be extended to include Atrial flutter. Longer term goal is to perform beat-level classification.

## Prerequisite

* [Python 3.9+](https://www.python.org)
* [Poetry](https://python-poetry.org/docs/#installation)

For `Apple Silicon based Macs` additional steps are required to properly install TensorFlow to leverage Metal. Please refer to [MacOS Setup](#TODO).

## Usage

To get started, first install the local python package `ecgarr` along with its dependencies via `Poetry`:

```bash
poetry install
```

The python package is intended to be used as a CLI-based app and provides a number of commands discussed below. In general, reference configurations are provided to (1) download dataset, (2) train model, (3) test model, (4) deploy model, and (5) run demo on Apollo 4 EVB hardware. Pre-trained Reference models are also included to enable running inference and the demo immediately.

### 1. Download Dataset (download_dataset)

This command will download the entire [Icentia11k dataset](https://physionet.org/content/icentia11k-continuous-ecg/1.0/) as a single zip file as well as convert into individual patient [HDF5 files](https://www.hdfgroup.org/solutions/hdf5/) (e.g. p00001.h5). The latter makes it possible to leverage Tensorflow datasets `prefetch` and `interleave` to parallelize loading data.

| _NOTE: The dataset requires roughly 300 GB of disk space and can take around 1 hour to download_ |

```bash
python -m ecgarr download_dataset --config-file ./configs/download_dataset.json
```

### 2. Train Model (train_model)

This command is used to train the arrhythmia model. The following command will train the model using the reference configuration. Please refer to `configs/train-model.json` and `ecgarr/types.py` to see supported options.

```bash
python -m ecgarr train_model --config-file ./configs/train-model.json
```

### 3. Test Model (test_model)

The following command will evaluate the accuracy of the model on the reserved test dataset.

```bash
python -m ecgarr test_model --config-file ./configs/test-model.json
```

### 4. Deploy (deploy_model)

The `Deploy command` will convert the referenced TensorFlow model into both TFLite and TFLite for microcontroller. The command will also verify the model outputs based on the dataset.

```bash
python -m ecgarr deploy_model --config-file ./configs/deploy-model.json
```

### 5. EVB Demo (evb_demo)

The 'EVB Demo command` is used to run the model on Ambiq Apollo 4 evaluation board (EVB). This setup requires both a host PC along with an Apollo 4 EVB. The host PC acts as a server and provides random samples to the EVB. The host PC is also used to provide status updates and model results from the EVB. The EVB runs in client mode- its job is to fetch samples and perform real-time inference using the arrhythmia model. Please refer to [EVB Demo Setup](#evb-demo-setup) for additional details.

```bash
python -m ecgarr evb_demo --config-file ./configs/evb-demo.json
```

## Architecture

The current network is based on a 1-D CNN architecture. The CNN is based on ResNet but adapted for 1-D input and utilizes longer filter sizes. The network as input takes windows (5 seconds) of raw ECG data. The only preprocessing performed is band-pass filtering and standardization on the window of ECG data.

## Datasets

A number of datasets are readily available online that are suitable for training heart arrhythmia models. The following datasets are ones we plan to explore. Currently, we are only using `Icentia11k` database as it contains the largest number of patients in a highly abulatory setting- users wearing a 1-lead chest band for up to two weeks. In future work, we intend to demonstrate fine-tuning the reference model on a different dataset (w/ different sensor(s)).

### Icentia 11k Database

This dataset consists of ECG recordings from 11,000 patients and 2 billion labelled beats. The data was collected by the CardioSTAT, a single-lead heart monitor device from Icentia. The raw signals were recorded with a 16-bit resolution and sampled at 250 Hz with the CardioSTAT in a modified lead 1 position. Please visit [Physionet](https://physionet.org/content/icentia11k-continuous-ecg/1.0/) for more details.

### MIT-BIH Arrhythmia Database

This dataset consists of ECG recordings from 47 different subjects recorded at a sampling rate of 360 Hz. 23 records (numbered from 100 to 124 inclusive with some numbers missing) chosen at random from this set, and 25 records (numbered from 200 to 234 inclusive, again with some numbers missing) selected from the same set to include a variety of rare but clinically important phenomena that would not be well-represented by a small random sample of Holter recordings. Each of the 48 records is slightly over 30 minutes long.

The subjects were 25 men aged 32 to 89 years, and 22 women aged 23 to 89 years. (Records 201 and 202 came from the same male subject.)

### MIT-BIH Normal Sinus Rhythm Database

This database includes 18 long-term ECG recordings of subjects referred to the Arrhythmia Laboratory at Boston's Beth Israel Hospital (now the Beth Israel Deaconess Medical Center). Subjects included in this database were found to have had no significant arrhythmias; they include 5 men, aged 26 to 45, and 13 women, aged 20 to 50.

### PTB Diagnostics dataset

Dataset consists of ECG records from 290 subjects: 148 diagnosed as MI, 52 healthy control, and the rest are diagnosed with 7 different disease. Each record contains ECG signals from 12 leads sampled at the frequency of 1000 Hz.

## EVB Demo Setup

* [Apollo4 EVB](https://ambiq.com/apollo4/)
* [MAX86150 Eval bard](https://protocentral.com/product/protocentral-max86150-ppg-and-ecg-breakout-with-qwiic-v2/)

## Hardware

### ECG / PPG ICs

* [AS7038RB](https://ams.com/en/as7038rb)
* [MAX86150](https://www.maximintegrated.com/en/products/interface/signal-integrity/MAX86150.html)

## Reference Papers

* [ECG Heartbeat classification using deep transfer learning with Convolutional Neural Network and STFT technique](https://arxiv.org/abs/2206.14200)
* [Classification of ECG based on Hybrid Features using CNNs for Wearable Applications](https://arxiv.org/pdf/2206.07648.pdf)
* [ECG Heartbeat classification using deep transfer learning with Convolutional Neural Network and STFT technique](https://arxiv.org/pdf/2206.14200.pdf)
