# Beat Classification Task

## <span class="sk-h2-span">Overview</span>

In beat classification, we classify individual beats as either normal, premature atrial contraction (PAC), or premature ventricular contraction (PVC).

<div class="sk-plotly-graph-div">
--8<-- "assets/pk_ecg_synthetic_afib.html"
</div>

---

## <span class="sk-h2-span">Characteristics</span>

| | Atrial | Junctional | Ventricular |
| --- | --- | --- | --- |
| Premature | __PAC__ <br> P-wave: Different <br> QRS: Narrow (normal) <br> Aberrated: LBBB or RBBB | __PJC__ <br> P-wave: None / retrograde <br> QRS: Narrow (normal) <br> Compensatory SA Pause | __PVC__ <br> P-wave: None <br> QRS: Wide (> 120 ms) <br> Compensatory SA PauseEscape |
| Atrial Escape | P-wave: Abnormal <br> QRS: Narrow (normal) <br> Ventricular rate: < 60 bpm <br> Junctional Escape <br> | P-wave: None <br> QRS: Narrow (normal) <br> Bradycardia (40-60 bpm) <br> Ventricular Escape | P-wave: None <br> QRS: Wide <br> Bradycardia (< 40 bpm) |

---

## <span class="sk-h2-span">Pre-trained Models</span>

The following table provides the latest performance and accuracy results for pre-trained beat models. Additional result details can be found in [Model Zoo → Beat](../zoo/beat.md).


--8<-- "assets/beat-model-zoo-table.md"

---

## <span class="sk-h2-span">Target Classes</span>

Below outlines the classes available for arrhythmia classification. When training a model, the number of classes, mapping, and names must be provided.

| CLASS   | LABELS          |
| ------- | --------------- |
| 0       | Normal          |
| 1       | PAC             |
| 2       | PVC             |
| 3       | Noise           |

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
