---
hide:
  - toc
---

# Model Zoo

A number of pre-trained models are available for download to use in your own project. These models are trained on the datasets listed below and are available in TensorFlow flatbuffer formats.

## <span class="sk-h2-span">[Denoising Task](./denoise.md)</span>

The following table provides the latest performance and accuracy results for denoising models.

--8<-- "assets/zoo/denoise/denoise-model-zoo-table.md"


## <span class="sk-h2-span">[Segmentation Task](./segmentation.md)</span>

The following table provides the latest performance and accuracy results for ECG segmentation models.

--8<-- "assets/zoo/segmentation/segmentation-model-zoo-table.md"

## <span class="sk-h2-span">[Rhythm Classification Task](./rhythm.md)</span>

The following table provides the latest performance and accuracy results for rhythm classification models.

--8<-- "assets/zoo/rhythm/rhythm-model-zoo-table.md"

## <span class="sk-h2-span">[Beat Classification Task](./beat.md)</span>

The following table provides the latest performance and accuracy results for beat classification models.

--8<-- "assets/zoo/beat/beat-model-zoo-table.md"

<!-- ## <span class="sk-h2-span">Multi-Label Diagnostic Classification</span>

The following table provides the latest performance and accuracy results for multi-label diagnostic classification models. Additional result details can be found in [Zoo → Diagnostic](./diagnostic.md).

--8<-- "assets/zoo/diagnostic/diagnostic-model-zoo-table.md" -->


## <span class="sk-h2-span"> Reproducing results </span>

Each pre-trained model has a corresponding `configuration.json` file that can be used to reproduce the model and results.

To reproduce a pre-trained rhythm model with configuration file `configuration.json`, run the following command:

```bash
heartkit -m train -t rhythm -c configuration.json
```

To evaluate the trained rhythm model with configuration file `configuration.json`, run the following command:

```bash
heartkit -m evaluate -t rhythm -c configuration.json
```
