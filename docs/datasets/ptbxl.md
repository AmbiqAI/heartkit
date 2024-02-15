# PTB Diagnostics Dataset

### <span class="sk-h2-span">Overview</span>

This dataset consists of ECG records from 290 subjects: 148 diagnosed as MI, 52 healthy controls, and the rest are diagnosed with 7 different disease. Each record contains ECG signals from 12 leads sampled at the frequency of 1000 Hz.

Please visit [Physionet](https://doi.org/10.13026/C28C71) for more details.

### <span class="sk-h2-span">Funding</span>

This work was supported by the German Federal Ministry of Education and Research (BMBF) within the framework of the e:Med research and funding concept (grant 01ZX1408A).

### <span class="sk-h2-span">License</span>

This database is available for commercial use.

### <span class="sk-h2-span">Supported Tasks</span>

* [Arrhythmia](../tasks/arrhythmia.md)

## <span class="sk-h2-span">Usage</span>

!!! Example Python

    ```python
    from pathlib import Path
    import heartkit as hk

    # Download dataset
    hk.datasets.download_datasets(hk.HKDownloadParams(
        ds_path=Path("./datasets"),
        datasets=["ptbxl"],
        progress=True
    ))
    ```
