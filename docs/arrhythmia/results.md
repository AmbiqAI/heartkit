# Results

## <span class="sk-h2-span">Overview</span>

The results of the arrhythmia models when testing on 1,000 patients (not used during training) is summarized below. The baseline model is simply selecting the argmax of model outputs (`normal`, `AFIB/AFL`). The 75% confidence version adds inconclusive label that is assigned when softmax output is less than 75% for any model output.

| Task           | Params   | FLOPS   | Metric     | Cycles/Inf | Time/Inf   |
| -------------- | -------- | ------- | ---------- | ---------- | ---------- |
| 2-Class        | 50K      | 3.6M    | 99.0% F1   | 89M        | 465ms      |
| 3-Class        | NA       | NA      | NA         | NA         | NA         |


## <span class="sk-h2-span">Metrics</span>

=== "2-Class"

    | Metric   | Baseline | 75% Confidence |
    | -------- | -------- | -------------- |
    | Accuracy | 96.5%    | 99.1%          |
    | F1 Score | 96.4%    | 99.0%          |
    | Drop     |  0.0%    | 12.0%          |

=== "3-Class"

    NA

## <span class="sk-h2-span">Confusion Matrices</span>

=== "2-Class"

    The confusion matrix for the 75% confidence model is depicted below.

    ![2-Stage Sleep Stage Confusion Matrix](../assets/arrhythmia-cm-test.png){ width="480" }

=== "3-Class"

    NA

## <span class="sk-h2-span">EVB Performance</span>

TODO

## <span class="sk-h2-span">Comparison</span>

TODO
