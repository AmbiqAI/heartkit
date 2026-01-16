# Beat Classification Task

## Overview

In beat classification, we classify individual beats as either normal or abnormal. Abnormal beats can be further classified as being either premature or escape beats as well as originating from the atria, junction, or ventricles. The objective of beat classification is to detect and classify these abnormal heart beats directly from ECG signals.

<div class="sk-plotly-graph-div">
--8<-- "assets/tasks/beat/beat-example.html"
</div>

---

## Characteristics

|     | Atrial | Junctional | Ventricular |
| --- | --- | --- | --- |
| Premature | __PAC__ <br> P-wave: Different <br> QRS: Narrow (normal) <br> Aberrated: LBBB or RBBB | __PJC__ <br> P-wave: None / retrograde <br> QRS: Narrow (normal) <br> Compensatory SA Pause | __PVC__ <br> P-wave: None <br> QRS: Wide (> 120 ms) <br> Compensatory SA Pause |
| Escape | Atrial Escape | P-wave: Abnormal <br> QRS: Narrow (normal) <br> Ventricular rate: < 60 bpm <br> Junctional Escape <br> | P-wave: None <br> QRS: Narrow (normal) <br> Bradycardia (40-60 bpm) <br> Ventricular Escape | P-wave: None <br> QRS: Wide <br> Bradycardia (< 40 bpm) |

---

## Dataloaders

Dataloaders are available for the following datasets:

* **[Icentia11k](../datasets/icentia11k.md)**
* **[PTB-XL](../datasets/ptbxl.md)**

---

## Pre-trained Models

The following table provides the latest performance and accuracy results for pre-trained beat models. Additional result details can be found in [Model Zoo → Beat](../zoo/beat.md).


--8<-- "assets/zoo/beat/beat-model-zoo-table.md"

---

## Target Classes

Below outlines the classes available for beat classification. When training a model, the number of classes, mapping, and names must be provided.

--8<-- "assets/tasks/beat/beat-classes.md"

!!! example "Class Mapping"

    Below is an example of a class mapping for a 3-class beat model. The class map keys are the original class labels and the values are the new class labels. Any class not included will be skipped.

    ```json
    {
        "num_classes": 3,
        "class_names": ["QRS", "PAC", "PVC"],
        "class_map": {
            "0": 0,  // Map Normal to QRS
            "1": 1,  // Map PAC to PAC
            "2": 2,  // Map PVC to PVC
        }
    }
    ```

---
