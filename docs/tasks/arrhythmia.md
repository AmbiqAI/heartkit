# Arrhythmia Classification Task

## <span class="sk-h2-span">Overview</span>

The objective of arrhythmia classification is to detect and classify abnormal heart rhythms. In particular, we focus on classifying abnormal heart rhythms such as atrial fibrillation (AFIB) and atrial flutter (AFL).

<div class="sk-plotly-graph-div">
--8<-- "assets/pk_ecg_synthetic_afib.html"
</div>

---

## <span class="sk-h2-span">Characteristics</span>

There are a variety of heart arrhythmias that can be detected using ECG signals. In this task, we focus on detecting rhythm-level arrhythmias such as atrial fibrillation (AFIB) and atrial flutter (AFL). The following table summarizes the characteristics of the included arrhythmias:

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

## <span class="sk-h2-span">Target Classes</span>

Below outlines the classes available for arrhythmia classification. When training a model, the number of classes, mapping, and names must be provided.

| CLASS   | LABELS          |
| ------- | --------------- |
| 0       | Normal          |
| 1       | AFIB            |
| 2       | AFL             |
| 3       | Noise           |


!!! example "Class Mapping"

    Below is an example of a class mapping for a 2-class arrhythmia model. The class map keys are the original class labels and the values are the new class labels. Any class not included will be skipped.

    ```json
    {
        "num_classes": 2,
        "class_names": ["NSR", "AFIB"],
        "class_map": {
            "0": 0,  // Map None to None
            "1": 1,  // Map AFIB to AFIB
            // Skip remaining classes
        }
    }
    ```

---

## <span class="sk-h2-span">References</span>

* [ECG Heartbeat classification using deep transfer learning with Convolutional Neural Network and STFT technique](https://arxiv.org/abs/2206.14200)
* [Classification of ECG based on Hybrid Features using CNNs for Wearable Applications](https://arxiv.org/pdf/2206.07648.pdf)
