# MIT BIH Normal Sinus Rhythm Dataset

## <span class="sk-h2-span">Overview</span>

This dataset includes 18 long-term ECG recordings of subjects referred to the Arrhythmia Laboratory at Boston's Beth Israel Hospital (now the Beth Israel Deaconess Medical Center). Subjects included in this dataset were found to have had no significant arrhythmias; they include 5 men, aged 26 to 45, and 13 women, aged 20 to 50.

Please visit [Physionet](https://doi.org/10.13026/C2NK5R) for more details.

## <span class="sk-h2-span">Funding</span>

N/A

## <span class="sk-h2-span">License</span>

This database is available for commercial use.

## <span class="sk-h2-span">Supported Tasks</span>

* [Arrhythmia](../tasks/arrhythmia.md)


## <span class="sk-h2-span">Usage</span>

!!! Example Python

    ```python
    from pathlib import Path
    import heartkit as hk

    # Download dataset
    hk.datasets.download_datasets(hk.HKDownloadParams(
        ds_path=Path("./datasets"),
        datasets=["mitbih"],
        progress=True
    ))
    ```