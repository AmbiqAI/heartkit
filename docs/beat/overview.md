# Beat Classification

## <span class="sk-h2-span">Overview</span>

In [Beat classification task](), we classify individual beats such as premature atrial contraction (PAC) and premature ventricular contraction (PVC).

<!-- <div class="sk-plotly-graph-div">
--8<-- "assets/pk_ecg_synthetic_afib.html"
</div> -->

## <span class="sk-h2-span">Characteristics</span>




| | Atrial | Junctional | Ventricular |
| --- | --- | --- | --- |
| Premature | PAC <br> * P-wave: Different <br> * QRS: Narrow (normal) <br> * Aberrated: LBBB or RBBB | PJC <br> P-wave: None / retrograde <br> QRS: Narrow (normal) <br> Compensatory SA Pause | PVC <br> P-wave: None <br> QRS: Wide (> 120 ms) <br> Compensatory SA PauseEscape |
| Atrial Escape | P-wave: Abnormal <br> QRS: Narrow (normal) <br> Ventricular rate: < 60 bpm <br> Junctional Escape <br> | P-wave: None <br> QRS: Narrow (normal) <br> Bradycardia (40-60 bpm) <br> Ventricular Escape | P-wave: None <br> QRS: Wide <br> Bradycardia (< 40 bpm) |
