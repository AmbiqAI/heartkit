# Architecture

HeartKit leverages a multi-head network- a backbone segmentation model followed by 3 upstream heads:

![HeartKit Architecture](./assets/heartkit-architecture.svg)

## ECG Segmentation

The ECG segmentation model serves as the backbone and is used to annotate every sample as either P-wave, QRS, T-wave, or none. The resulting ECG data and segmentation mask is then fed into upstream “heads”. This model utilizes a custom 1-D UNET architecture w/ additional skip connections between encoder and decoder blocks. The encoder blocks are convolutional based and include both expansion and inverted residuals layers. The only preprocessing performed is band-pass filtering and standardization on the window of ECG data.

## HRV Head

The HRV head uses only DSP and statistics (i.e. no network is used). The segmentation results are stitched together and used to derive several useful metrics including heart rate, rhythm and RR interval.


## Arrhythmia Head

The arrhythmia head is used to detect the presence of Atrial Fibrillation (AFIB) or Atrial Flutter (AFL). Note that if heart arrhythmia is detected, the remaining heads are skipped. The arrhythmia model utilizes a 1-D CNN built using MBConv style blocks that incorporate expansion, inverted residuals, and squeeze and excitation layers. Furthermore, longer filter and stide lengths are utilized in the initial layers to capture more temporal dependencies.


## Beat Head

The beat head is used to extract individual beats and classify them as either normal, premature/ectopic atrial contraction (PAC), premature/ectopic ventricular contraction (PVC), or noise. In addition to the target beat, the surrounding beats are also fed into the network as context. The “neighboring” beats are determined based on the average RR interval and not the actual R peak. The beat head also utilizes a 1-D CNN built using MBConv style blocks.
