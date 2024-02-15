# Arrhythmia Classification

## <span class="sk-h2-span">Overview</span>

The objective of arrhythmia classification is to detect and classify abnormal heart rhythms. In particular, we focus on classifying abnormal heart rhythms such as atrial fibrillation (AFIB) and atrial flutter (AFL).

<div class="sk-plotly-graph-div">
--8<-- "assets/pk_ecg_synthetic_afib.html"
</div>

---

## <span class="sk-h2-span">Characteristics</span>

There are a variety of heart arrhythmias that can be detected using ECG signals. In this task, we focus on detecting and classifying atrial fibrillation (AFIB) and atrial flutter (AFL). The following table summarizes the characteristics of AFIB and AFL:

=== "AFIB"

    Atrial fibrillation (AFIB) is a type of arrhythmia where the atria (upper chambers of the heart) beat irregularly and out of sync with the ventricles (lower chambers of the heart). AFIB is the most common type of arrhythmia and can lead to serious complications such as stroke and heart failure. AFIB is typically characterized by the following:

    * Irregularly irregular rhythm
    * No P waves
    * Variable ventricular rate
    * QRS complexes usually < 120ms
    * Fibrillatory waves may be present

=== "AFL"

    Atrial flutter (AFL) is a type of arrhythmia where the atria (upper chambers of the heart) beat regularly but faster than normal. AFL is less common than AFIB and can lead to serious complications such as stroke and heart failure. AFL is typically characterized by the following:

    * Narrow complex tachycardia
    * Regular atrial activity at ~300 bpm
    * Loss of the isoelectric baseline
    * “Saw-tooth” pattern of inverted flutter waves in leads II, III, aVF
    * Upright flutter waves in V1 that may resemble P waves
    * Ventricular rate depends on AV conduction ratio

---

## <span class="sk-h2-span">Pre-trained Models</span>

The following table provides the latest performance and accuracy results for arrhythmia models. Additional result details can be found in [Model Zoo → Arrhythmia](../zoo/arrhythmia.md).


--8<-- "assets/arrhythmia-model-zoo-table.md"

---

## <span class="sk-h2-span">Classes</span>

Below outlines the class labels used for arrhythmia classification.

=== "2-Class"

    | CLASS    | LABELS           |
    | -------- | ---------------- |
    | 0        | NSR              |
    | 1        | AFIB, AFL        |

=== "3-Class"

    | CLASS   | LABELS           |
    | ------- | ---------------- |
    | 0       | NSR              |
    | 1       | AFIB             |
    | 2       | AFL              |

---


## <span class="sk-h2-span">References</span>

* [ECG Heartbeat classification using deep transfer learning with Convolutional Neural Network and STFT technique](https://arxiv.org/abs/2206.14200)
* [Classification of ECG based on Hybrid Features using CNNs for Wearable Applications](https://arxiv.org/pdf/2206.07648.pdf)
* [ECG Heartbeat classification using deep transfer learning with Convolutional Neural Network and STFT technique](https://arxiv.org/pdf/2206.14200.pdf)
* [U2-Net: Going Deeper with Nested U-Structure for Salient Object Detection](https://arxiv.org/abs/2005.09007)
* [UNET 3+: A FULL-SCALE CONNECTED UNET FOR MEDICAL IMAGE SEGMENTATION](https://arxiv.org/pdf/2004.08790.pdf)
* [ResUNet-a: a deep learning framework for semantic segmentation of remotely sensed data](https://arxiv.org/pdf/1904.00592.pdf)
