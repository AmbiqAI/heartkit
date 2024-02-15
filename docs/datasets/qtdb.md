# QTDB Dataset

## <span class="sk-h2-span">Overview</span>

Over 100 fifteen-minute two-lead ECG recordings with onset, peak, and end markers for P, QRS, T, and (where present) U waves of from 30 to 50 selected beats in each recording.

Please visit [Physionet](https://doi.org/10.13026/C24K53) for more details.

## <span class="sk-h2-span">Funding</span>

The QT Database was created as part of a project funded by the National Library of Medicine.

## <span class="sk-h2-span">License</span>

The QT Database is available for commercial use.

## <span class="sk-h2-span">Supported Tasks</span>

* [Segmentation](../tasks/segmentation.md)

## <span class="sk-h2-span">Usage</span>

!!! Example Python

    ```python
    from pathlib import Path
    import heartkit as hk

    # Download dataset
    hk.datasets.download_datasets(hk.HKDownloadParams(
        ds_path=Path("./datasets"),
        datasets=["qtdb"],
        progress=True
    ))
    ```
