# Model Zoo

A number of pre-trained models are available for download to use in your own project. These models are trained on the datasets listed below and are available in Keras and TensorFlow Lite flatbuffer formats.

## <span class="sk-h2-span">[Signal Denoising Task](../tasks/denoise.md)</span>

The following table provides the latest performance and accuracy results for denoising models.

| NAME                | DATASET           | FS    | DURATION | MODEL          | PARAMS | FLOPS   | METRIC      |
| ------------------- | ----------------- | ----- | -------- | -------------- | ------ | ------- | ----------- |
| __DEN-TCN-SM__      | Synthetic, PTB-XL | 100Hz | 2.5s     | TCN            | 3.3K   | 1.0M    | 18.1 SNR    |
| __DEN-TCN-LG__      | Synthetic, PTB-XL | 100Hz | 2.5s     | TCN            | 6.3K   | 1.8M    | 19.5 SNR    |
| __DEN-PPG-TCN-SM__  | Synthetic         | 100Hz | 2.5s     | TCN            | 3.5K   | 1.1M    | 92.1% COS   |


## <span class="sk-h2-span">[Signal Segmentation Task](../tasks/segmentation.md)</span>

The following table provides the latest performance and accuracy results for ECG segmentation models.

| NAME                 | DATASET                  | FS    | DURATION | # CLASSES | MODEL         | PARAMS | FLOPS   | METRIC    |
| -------------------- | ------------------------ | ----- | -------- | --------- | ------------- | ------ | ------- | --------- |
| __SEG-2-TCN-SM__     | LUDB, Synthetic          | 100Hz | 2.5s     | 2         | TCN           | 2K     | 0.42M   | 96.6% F1  |
| __SEG-4-TCN-SM__     | LUDB, Synthetic          | 100Hz | 2.5s     | 4         | TCN           | 7K     | 2.1M    | 86.3% F1  |
| __SEG-4-TCN-LG__     | LUDB, Synthetic          | 100Hz | 2.5s     | 4         | TCN           | 10K    | 3.9M    | 89.4% F1  |
| __SEG-PPG-2-TCN-SM__ | Synthetic                | 100Hz | 2.5s     | 2         | TCN           | 4K     | 1.43M   | 98.6% F1  |


## <span class="sk-h2-span">[Rhythm Classification Task](../tasks/rhythm.md)</span>

The following table provides the latest performance and accuracy results for rhythm classification models.

| NAME             | DATASET                  | FS    | DURATION | # CLASSES | MODEL          | PARAMS | FLOPS   | METRIC   |
| ---------------- | ------------------------ | ----- | -------- | --------- | -------------- | ------ | ------- | -------- |
| __ARR-2-EFF-SM__ | Icentia11K, PTB-XL, LSAD | 100Hz | 5s       | 2         | EfficientNetV2 | 18K    |  1.2M   | 99.5% F1 |
| __ARR-4-EFF-SM__ | LSAD                     | 100Hz | 5s       | 4         | EfficientNetV2 | 27K    |  1.6M   | 95.9% F1 |


## <span class="sk-h2-span">[Beat Classification Task](../tasks/beat.md)</span>

The following table provides the latest performance and accuracy results for beat classification models.

| NAME            | DATASET    | FS    | DURATION | # CLASSES | MODEL          | PARAMS | FLOPS   | METRIC   |
| --------------- | ---------- | ----- | -------- | --------- | -------------- | ------ | ------- | -------- |
| __BC-2-EFF-SM__ | Icentia11k | 100Hz | 5s       | 2         | EfficientNetV2 | 28K    | 1.8M    | 97.7% F1 |
| __BC-3-EFF-SM__ | Icentia11k | 100Hz | 5s       | 3         | EfficientNetV2 | 41K    | 2.1M    | 92.0% F1 |


## <span class="sk-h2-span"> Reproducing Results </span>

Each pre-trained model has a corresponding `configuration.json` file that can be used to reproduce the model and results.

To reproduce a pre-trained rhythm model with configuration file `configuration.json`, run the following command:

```bash
heartkit -m train -t rhythm -c configuration.json
```

To evaluate the trained rhythm model with configuration file `configuration.json`, run the following command:

```bash
heartkit -m evaluate -t rhythm -c configuration.json
```
