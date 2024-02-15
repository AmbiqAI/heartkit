# ECG Segmentation Task

## <span class="sk-h2-span">Overview</span>

The objective of ECG segmentation is to delineate key segments of the cardiac cycle, including the P-wave, QRS complex, and T-wave. These segments are used to compute a number of clinically relevant metrics, including heart rate, PR interval, QRS duration, QT interval, and QTc interval. They are also useful for a variety of upstream tasks, including arrhythmia classification and beat classification.


<div class="sk-plotly-graph-div">
--8<-- "assets/pk_ecg_synthetic_nsr.html"
</div>

## <span class="sk-h2-span">Characteristics</span>


* __P-Wave__: Reflects atrial depolarization
* __QRS Complex__: Reflects ventricular depolarization
* __T-Wave__: Reflects ventricular repolarization
* __U-Wave__: Reflects papillary muscle repolarization


<figure markdown>
  ![Annotated ECG Signal](../assets/ecg-annotated.svg){ width="380" }
  <figcaption>Annotated ECG Signal</figcaption>
</figure>

## <span class="sk-h2-span">Pre-Trained Models</span>

The following table provides the latest performance and accuracy results for ECG segmentation models. Additional result details can be found in [Model Zoo → Segmentation](../zoo/segmentation.md).

--8<-- "assets/segmentation-model-zoo-table.md"

## <span class="sk-h2-span">Target Classes</span>

Below outlines the class labels available for ECG segmentation. When training a model, the number of classes can be specified based on the desired level of granularity.

=== "2-Class"

    Only detect QRS complexes.

    | CLASS    | LABELS                |
    | -------- | --------------------- |
    | 0        | None, P-wave, T-wave  |
    | 1        | QRS                   |

=== "3-Class"

    Bucket the P-wave and T-wave into a single class.

    | CLASS   | LABELS          |
    | ------- | --------------- |
    | 0       | None            |
    | 1       | QRS             |
    | 2       | P-wave, T-wave  |

=== "4-Class"

    Identify each of the P-wave, QRS complex, and T-wave.

    | CLASS   | LABELS          |
    | ------- | --------------- |
    | 0       | None            |
    | 1       | P-wave          |
    | 2       | QRS             |
    | 3       | T-wave          |

=== "5-Class"

    Same as 4-Class, but also detect U-waves.

    | CLASS   | LABELS          |
    | ------- | --------------- |
    | 0       | None            |
    | 1       | P-wave          |
    | 2       | QRS             |
    | 3       | T-wave          |
    | 4       | U-wave          |
