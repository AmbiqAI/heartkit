# Architecture

HeartKit leverages a multi-head network- a backbone segmentation model followed by 3 upstream heads:

![HeartKit Architecture](./assets/heartkit-architecture.svg)

## ECG Segmentation

The ECG segmentation model serves as the backbone and is used to annotate every sample as either P-wave, QRS, T-wave, or none. The resulting ECG data and segmentation mask is then fed into upstream “heads”. This model utilizes a custom 1-D UNET architecture w/ additional skip connections between encoder and decoder blocks. The encoder blocks are convolutional based and include both expansion and inverted residuals layers. The only preprocessing performed is band-pass filtering and standardization on the window of ECG data.

## HRV Head

The HRV head uses only DSP and statistics (i.e. no neural network is used). Using a combination of segmentation results and QRS filter, the HRV head detects R peak candidates. RR intervals are extracted and filtered, and then used to derive a variety of HRV metrics including heart rate, rhythm, SDNN, SDRR, SDANN, etc. All of the identified R peaks are further fed to the beat classifier head. Note that if segmentation model is not enabled, HRV head falls back to identifying R peaks purely on gradient of QRS signal.

## Arrhythmia Head

The arrhythmia head is used to detect the presence of Atrial Fibrillation (AFIB) or Atrial Flutter (AFL). Note that if heart arrhythmia is detected, the remaining heads are skipped. The arrhythmia model utilizes a 1-D CNN built using MBConv style blocks that incorporate expansion, inverted residuals, and squeeze and excitation layers. Furthermore, longer filter and stide lengths are utilized in the initial layers to capture more temporal dependencies.

## Beat Head

The beat head is used to extract individual beats and classify them as either normal, premature/ectopic atrial contraction (PAC), premature/ectopic ventricular contraction (PVC), or noise. In addition to the target beat, the surrounding beats are also fed into the network as context. The “neighboring” beats are determined based on the average RR interval and not the actual R peak. The beat head also utilizes a 1-D CNN built using MBConv style blocks.
