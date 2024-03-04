# Rhythm Classification Task

## <span class="sk-h2-span">Overview</span>

The objective of rhythm classification is to detect and classify abnormal heart rhythms, also known as arrhythmias, directly from ECG signals.

<div class="sk-plotly-graph-div">
--8<-- "assets/pk_ecg_synthetic_afib.html"
</div>

---

## <span class="sk-h2-span">Characteristics</span>

There are a variety of heart rhythms that can be detected using ECG signals. In this task, we predominantly focus on detecting arrhythmias such as atrial fibrillation (AFIB) and atrial flutter (AFL). The following table summarizes characteristics of a few rhythms:


=== "SR"

    Sinus rhythm (SR) is a type of rhythm where the heart beats regularly and originates from the sinus node. SR is considered normal and is typically characterized by the following:

    * Regular rhythm
    * P waves present
    * QRS complexes usually < 120ms
    * Ventricular rate 60-100 bpm

=== "AFIB"

    Atrial fibrillation (AFIB) is a type of rhythm where the atria (upper chambers of the heart) beat irregularly and out of sync with the ventricles (lower chambers of the heart). AFIB is the most common type of rhythm and can lead to serious complications such as stroke and heart failure. AFIB is typically characterized by the following:

    * Irregularly irregular rhythm
    * No P waves
    * Variable ventricular rate
    * QRS complexes usually < 120ms
    * Fibrillatory waves may be present

=== "AFL"

    Atrial flutter (AFL) is a type of rhythm where the atria (upper chambers of the heart) beat regularly but faster than normal. AFL is less common than AFIB and can lead to serious complications such as stroke and heart failure. AFL is typically characterized by the following:

    * Narrow complex tachycardia
    * Regular atrial activity at ~300 bpm
    * Loss of the isoelectric baseline
    * “Saw-tooth” pattern of inverted flutter waves in leads II, III, aVF
    * Upright flutter waves in V1 that may resemble P waves
    * Ventricular rate depends on AV conduction ratio

=== "SVT"

    Supraventricular tachycardia (SVT) is a type of rhythm where the heart beats faster than normal due to abnormal electrical signals originating above the ventricles. SVT is typically characterized by the following:

    * Narrow complex tachycardia
    * Regular rhythm
    * P waves may be absent, hidden, or retrograde
    * QRS complexes usually < 120ms
    * Ventricular rate > 100 bpm

=== "BRAD"

    Sinus bradycardia (BRAD) is a type of rhythm where the heart beats slower than normal due to abnormal electrical signals originating from the sinus node. BRAD is typically characterized by the following:

    * Regular rhythm
    * P waves present
    * QRS complexes usually < 120ms
    * Ventricular rate < 60 bpm

---

## <span class="sk-h2-span">Pre-trained Models</span>

The following table provides the latest performance and accuracy results for rhythm models. Additional result details can be found in [Model Zoo → Rhythm](../zoo/rhythm.md).


--8<-- "assets/rhythm-model-zoo-table.md"

---

## <span class="sk-h2-span">Target Classes</span>

Below outlines the classes available for rhythm classification. When training a model, the number of classes, mapping, and names must be provided.

| CLASS   | LABEL | DESCRIPTION |
| ------- | ----- | ----------- |
| NSR     | 0     | Normal sinus rhythm |
| SBRAD   | 1     | Sinus bradycardia |
| STACH   | 2     | Sinus tachycardia |
| SARRH   | 3     | Sinus arrhythmia |
| SVARR   | 4     | Supraventricular arrhythmia |
| SVT     | 5     | Supraventricular tachycardia |
| VTACH   | 6     | Ventricular tachycardia |
| AFIB    | 7     | Atrial fibrillation |
| AFLUT   | 8     | Atrial flutter |
| VFIB    | 9     | Ventricular fibrillation |
| VFLUT   | 10    | Ventricular flutter |
| BIGU    | 11    | Bigeminy (every other beat is PVC) |
| TRIGU   | 12    | Trigeminy (every third beat is PVC) |
| PACE    | 13    | Paced rhythm |
| NOISE   | 127   | Noise |


!!! example "Class Mapping"

    Below is an example of a class mapping for a 2-class rhythm model. The class map keys are the original class labels and the values are the new class labels. Any class not included will be skipped.

    ```json
    {
        "num_classes": 2,
        "class_names": ["NSR", "AFIB"],
        "class_map": {
            "0": 0,  // Map None to None
            "7": 1,  // Map AFIB to AFIB
            // Skip remaining classes
        }
    }
    ```

---

## <span class="sk-h2-span">References</span>

* [ECG Heartbeat classification using deep transfer learning with Convolutional Neural Network and STFT technique](https://arxiv.org/abs/2206.14200)
* [Classification of ECG based on Hybrid Features using CNNs for Wearable Applications](https://arxiv.org/pdf/2206.07648.pdf)
