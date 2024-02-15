# LUDB Dataset

## <span class="sk-h2-span">Overview</span>

The Lobachevsky University Electrocardiography database (LUDB) consists of 200 10-second 12-lead records. The boundaries and peaks of P, T waves and QRS complexes were manually annotated by cardiologists. Each record is annotated with the corresponding diagnosis.

Please visit [Physionet](https://physionet.org/content/ludb/1.0.1/) for more details.

## <span class="sk-h2-span">Funding</span>

The study was supported by the Ministry of Education of the Russian Federation (contract No. 02.G25.31.0157 of 01.12.2015).

## <span class="sk-h2-span">License</span>

The LUDB is available for commercial use.

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
        datasets=["ludb"],
        progress=True
    ))
    ```
