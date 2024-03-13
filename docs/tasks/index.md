# Tasks

## <span class="sk-h2-span">Introduction</span>

HeartKit provides several built-in __heart-monitoring__ related tasks. Each task is designed to address a unique aspect such as ECG denoising, segmentation, and rhythm/beat classification. The tasks are designed to be modular and can be used independently or in combination to address specific use cases. In addition to the built-in tasks, custom tasks can be created by extending the `HKTask` base class and registering it with the task factory.

## <span class="sk-h2-span">Available Tasks</span>

### <span class="sk-h2-span"> [Denoise](./denoise.md)</span>

ECG denoising is the process of removing noise from an ECG signal. This task is useful for improving the quality of the ECG signal and for further downstream tasks such as segmentation.

### <span class="sk-h2-span">[Segmentation](./segmentation.md)</span>

ECG segmentation is the process of delineating an ECG signal into individual waves (e.g. P-wave, QRS, T-wave). This task is useful for extracting features (e.g. HRV) from the ECG signal and for further analysis such as rhythm classification.

### <span class="sk-h2-span">[Rhythm](./rhythm.md)</span>

Rhythm classification is the process of identifying abnormal heart rhythms, also known as arrhythmias, such as atrial fibrillation (AFIB) and atrial flutter (AFL). Cardiovascular diseases such as AFIB are a leading cause of morbidity and mortality worldwide. Being able to remotely identify heart arrhtyhmias is important for early detection and intervention.

### <span class="sk-h2-span">[Beat](./beat.md)</span>

Beat classification is the process of identifying and classifying individual heart beats such as normal, premature, and escape beats. By identifying abnormal heart beats, it is possible to detect and monitor various heart conditions.

### <span class="sk-h2-span">[Diagnostic](./diagnostic.md)</span>

Multi-label diagnostic classification is the process of assigning diagnostic labels to an ECG signal. The diagnostic labels are structured in a hierarchical organization in terms of 5 coarse superclasses and 24 subclasses.

### <span class="sk-h2-span">[Bring-Your-Own-Task (BYOT)](./byot.md)</span>

Bring-Your-Own-Task (BYOT) is a feature that allows users to create custom tasks by extending the `HKTask` base class and registering it with the task factory. This feature is useful for addressing specific use cases that are not covered by the built-in tasks.

---

!!! Example "Recap"

    === "Denoise"

        ### ECG Denoising

        Remove noise from ECG signal. <br>
        Refer to [Denoise Task](./denoise.md) for more details.

    === "Segmentation"

        ### ECG Segmentation

        Delineate ECG signal into individual waves (e.g. P-wave, QRS, T-wave). <br>
        Refer to [Segmentation Task](./segmentation.md) for more details.

    === "Rhythm"

        ### Rhythm Classification

        Identify rhythm-level arrhythmias such as AFIB and AFL. <br>
        Refer to [Rhythm Task](./rhythm.md) for more details.


    === "Beat"

        ### Beat Classification

        Identify premature and escape beats. <br>
        Refer to [Beat Task](./beat.md) for more details.

    === "Diagnostic"

        ### Diagnostic Classification

        Assign diagnostic labels to an ECG signal. <br>
        Refer to [Diagnostic Task](./diagnostic.md) for more details.

---
