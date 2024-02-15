# Tasks

## <span class="sk-h2-span">Introduction</span>

HeartKit provides several built-in __heart-monitoring__ related tasks. Each task is designed to address a unique aspect such as ECG segmentation, arrhythmia classification, and beat classification. The tasks are designed to be modular and can be used independently or in combination to address specific use cases. In addition to the built-in tasks, custom tasks can be created by extending the base task class and registering it with the task factory.

## <span class="sk-h2-span">[Segmentation](./segmentation.md)</span>

ECG segmentation is the process of delineating an ECG signal into individual waves (e.g. P-wave, QRS, T-wave). This task is useful for extracting features from the ECG signal and for further analysis such as arrhythmia classification.

## <span class="sk-h2-span">[Arrhythmia](./arrhythmia.md)</span>

Arrhythmia classification is the process of identifying rhythm-level arrhythmias such as AFIB and AFL. This task is useful for identifying abnormal heart rhythms and for further analysis such as beat classification.

## <span class="sk-h2-span">[Beat](./beat.md)</span>

Beat classification is the process of identifying premature and escape beats. This task is useful for identifying abnormal heart beats.

## <span class="sk-h2-span"> [Denoise](./denoise.md)</span>

ECG denoising is the process of removing noise from an ECG signal. This task is useful for improving the quality of the ECG signal and for further analysis such as arrhythmia classification.

## <span class="sk-h2-span">[Bring-Your-Own-Task (BYOT)](./byot.md)</span>

Bring-Your-Own-Task (BYOT) is a feature that allows users to create custom tasks by extending the base task class and registering it with the task factory. This feature is useful for addressing specific use cases that are not covered by the built-in tasks.

!!! Example "At-a-Glance"

    === "Segmentation"

        ### ECG Segmentation

        Delineate ECG signal into individual waves (e.g. P-wave, QRS, T-wave). <br>
        Refer to [Segmentation Task](./segmentation.md) for more details.

    === "Arrhythmia"

        ### Arrhythmia Classification

        Identify rhythm-level arrhythmias such as AFIB and AFL. <br>
        Refer to [Arrhythmia Task](./arrhythmia.md) for more details.


    === "Beat"

        ### Beat Classification

        Identify premature and escape beats. <br>
        Refer to [Beat Task](./beat.md) for more details.

    === "Denoise"

        ### ECG Denoising

        Remove noise from ECG signal. <br>
        Refer to [Denoise Task](./denoise.md) for more details.

---
