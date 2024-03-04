# ECG Denoising

## <span class="sk-h2-span">Overview</span>

The objective of ECG denoising is to remove noise and artifacts from ECG signals while preserving the underlying cardiac information. The dominant noise sources include baseline wander (BW), muscle noise (EMG), electrode movement artifacts (EM), and powerline interference (PLI). While traditional signal processing techniques such as filtering and wavelet denoising have been used to remove noise, deep learning models have shown great promise in enhanced ECG denoising.

<div class="sk-plotly-graph-div">
--8<-- "assets/denoise_example.html"
</div>

---

## <span class="sk-h2-span">Noise Characteristics</span>

The following table summarizes the characteristics of common noise sources in ECG signals:

| Type | Causes | Spectrum | Effects |
| --- | --- | --- | --- |
| Baseline Wander (BW) | Respiration, posture changes | 0-1.0 Hz | Distorts ST segment and other LF components |
| Powerline Interference (PLI) | Electrical equipment | 50-60 Hz | Distorts P and T waves |
| Muscle Noise (EMG) | Muscle activity | 0-100 Hz | Distorts local waves |
| Electrode Movement (EM) | Electrode motion, skinimpedance | 0-100 Hz | Distorts local waves |

---

## <span class="sk-h2-span">Pre-trained Models</span>

The following table provides the latest performance and accuracy results of denoising models. Additional result details can be found in [Model Zoo → Denoise](../zoo/denoise.md).


--8<-- "assets/denoise-model-zoo-table.md"

---

## <span class="sk-h2-span">References</span>

* [DeScoD-ECG: Deep Score-Based Diffusion Model for ECG Baseline Wander and Noise Removal](https://arxiv.org/pdf/2208.00542.pdf)
* [DEEPFILTER: AN ECG BASELINE WANDER REMOVAL FILTER USING DEEP LEARNING TECHNIQUES](https://arxiv.org/pdf/2101.03423.pdf)
