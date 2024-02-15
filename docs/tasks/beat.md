# Beat Classification

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

## <span class="sk-h2-span">Classes</span>

Below outlines the class labels used for beat classification.

=== "2-Stage"

    | CLASS    | LABELS           |
    | -------- | ---------------- |
    | 0        | NSR              |
    | 1        | PAC/PVC          |

=== "3-Stage"

    | CLASS   | LABELS           |
    | ------- | ---------------- |
    | 0       | NSR              |
    | 1       | PAC              |
    | 2       | PVC              |

---
