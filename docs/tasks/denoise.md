# Signal Denoising Task

## <span class="sk-h2-span">Overview</span>

The objective of denoising is to remove noise and artifacts from physiological signals while preserving the underlying signal information. The dominant noise sources include baseline wander (BW), muscle noise (EMG), electrode movement artifacts (EM), and powerline interference (PLI). For physiological signals such as ECG and PPG, removing the artifacts is difficult due to the non-stationary nature of the noise and overlapping frequency bands with the signal. While traditional signal processing techniques such as filtering and wavelet denoising have been used to remove noise, deep learning models have shown great promise in enhanced denoising.

<div class="sk-plotly-graph-div">
--8<-- "assets/tasks/denoise/denoise-example.html"
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

The following table summarizes the characteristics of common noise sources in PPG signals:

| Type | Causes | Spectrum | Effects |
| --- | --- | --- | --- |
| Motion Artifacts | Movement, pressure | 0-10 Hz | Distorts signal |
| Ambient Light | Sunlight, artificial light | 0-100 Hz | Distorts signal |
| Blood Pressure | Blood flow, pressure | 0-10 Hz | Distorts signal |

---

## <span class="sk-h2-span">Dataloaders</span>

Dataloaders are available for the following datasets:

* **[LUDB](../datasets/ludb.md)**
* **[PTB-XL](../datasets/ptbxl.md)**
* **[ECG Synthetic](../datasets/synthetic.md)**
* **[PPG Synthetic](../datasets/synthetic.md)**

---

## <span class="sk-h2-span">Pre-trained Models</span>

The following table provides the latest performance and accuracy results of denoising models. Additional result details can be found in [Model Zoo → Denoise](../zoo/denoise.md).


--8<-- "assets/zoo/denoise/denoise-model-zoo-table.md"

---

## <span class="sk-h2-span">References</span>

* [DeScoD-ECG: Deep Score-Based Diffusion Model for ECG Baseline Wander and Noise Removal](https://arxiv.org/pdf/2208.00542.pdf)
* [DEEPFILTER: AN ECG BASELINE WANDER REMOVAL FILTER USING DEEP LEARNING TECHNIQUES](https://arxiv.org/pdf/2101.03423.pdf)
