# Arrhythmia Classification

## <span class="sk-h2-span">Overview</span>

The objective of arrhythmia classification is to detect and classify abnormal heartbeat. In particular, we focus on classifying abnormal heart rhythms such as atrial fibrillation (AFIB) and atrial flutter (AFL).

<div class="sk-plotly-graph-div">
--8<-- "assets/pk_ecg_synthetic_afib.html"
</div>

## <span class="sk-h2-span">Characteristics</span>

There are a variety of heart arrhythmias that can be detected using ECG signals. In this task, we focus on detecting and classifying atrial fibrillation (AFIB) and atrial flutter (AFL). The following table summarizes the characteristics of AFIB and AFL:

=== "AFIB"

    Atrial fibrillation (AFIB) is a type of arrhythmia where the atria (upper chambers of the heart) beat irregularly and out of sync with the ventricles (lower chambers of the heart). AFIB is the most common type of arrhythmia and can lead to serious complications such as stroke and heart failure. AFIB is typically characterized by the following:

    * Re-entrant circuit is present
    * Atrium depolarization is 250-350 bpm
    * Characteristic sawtooth pattern in P waves (f waves)
    * QRS: Narrow (normal)
    * AV conduction ratio (2:1, 3:1)

=== "AFL"

    Atrial flutter (AFL) is a type of arrhythmia where the atria (upper chambers of the heart) beat regularly but faster than normal. AFL is less common than AFIB and can lead to serious complications such as stroke and heart failure. AFL is typically characterized by the following:

    * AP fire chaotically within pulmonary veins / atrium (quiver)
    * Atrium depolarization is 400-600 bpm (ventricular 100-200 bpm)
    * AV node is intermittently refractory
    * Varying RR intervals
