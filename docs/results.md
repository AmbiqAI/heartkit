# Results

## Overview

The following table provides performance and accuracy results of all models when running on Apollo 4 EVB.

| Task           | Params   | FLOPS   | Metric      |
| -------------- | -------- | ------- | ----------- |
| Segmentation   | 105K     | 19.3M   | IOU=85.3%   |
| Arrhythmia     | 76K      | 7.2M    | F1=99.4%    |
| Beat           | 79K      | 1.6M    | F1=91.6%    |
| HRV            | N/A      | N/A     | N/A         |

## Segmentation Results

Work in progress...

## Heart Arrhythmia Results

The results of the arrhythmia model when testing on 1,000 patients (not used during training) is summarized below. The baseline model is simply selecting the argmax of model outputs (`normal`, `AFIB/AFL`). The 95% confidence version adds inconclusive label that is assigned when softmax output is less than 95% for any model output.

| Metric   | Baseline | 95% Confidence |
| -------- | -------- | -------------- |
| Accuracy | 96.2%   | 99.4%           |
| F1 Score | 96.2%   | 99.4%           |

The confusion matrix for the 95% confidence model is depicted below.

| Confusion    | NSR      | AFIB/AFL |
| ------------ | -------- | -------- |
| __NSR__      | 99.5%    |  0.5%    |
| __AFIB/AFL__ |  0.7%    | 99.3%    |

## Beat Classification Results

The results of three beat models when testing on 1,000 patients (not used during training) are summarized below. The 200x1 model serves as the baseline and classifies individual beats (1 channel) with a fixed time window of 800 ms (200 samples). The 800x1 model increases the time window to 3,200 ms (800 samples) in order to include surrounding data as context. Increasing the time window increases the accuracy by over `10%` but also causes computation to increase by `3.5x`. The third and best model uses a time window of 800 ms to capture individual beats but includes two additional channels. Using the local average RR interval, the previous and subsequent `beats` are included as side channels. Unlike normal beats, premature and ectopic beats won't be aligned to neighboring beats and serves as useful context. This provides similar temporal resolution as 800x1 but reduces computation by `3.3x` while further improving accuracy by `1.7%`.

| Model      | 200x1  | 800x1  | 200x3  |
| ---------- | ------ | ------ | ------ |
| Parameters | 79K    | 79K    | 79K    |
| FLOPS      | 1.5M   | 5.3M   | 1.6M   |
| Accuracy   | 78.2%  | 88.6%  | 90.3%  |
| F1 Score   | 77.5%  | 87.2%  | 90.1%  |

The confusion matrix for the 200x3 model is depicted below.

| Confusion | Normal | PAC   | PVC   |
| --------- | ------ | ----- | ----- |
| __NSR__   | 94.6%  |  4.6% |  0.8% |
| __PAC__   |  4.9%  | 86.5% |  8.6% |
| __PVC__   |  0.7%  | 10.2% | 89.0% |

## HRV Results

The HRV metrics are computed using off-the-shelf definitions based purely on the output of the segmentation and beat models. The current metrics include heart rate, rhythm, and RR variation. We intend to include additional metrics later such as QTc along with frequency metrics.
