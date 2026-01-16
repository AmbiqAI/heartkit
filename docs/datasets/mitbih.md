# MIT BIH Normal Sinus Rhythm Dataset

## Overview

This dataset includes 18 long-term ECG recordings of subjects referred to the Rhythm Laboratory at Boston's Beth Israel Hospital (now the Beth Israel Deaconess Medical Center). Subjects included in this dataset were found to have had no significant arrhythmias; they include 5 men, aged 26 to 45, and 13 women, aged 20 to 50.

Please visit [Physionet](https://doi.org/10.13026/C2NK5R) for more details.

## Funding

N/A

## License

This database is available for commercial use. [Open Data Commons Attribution License v1.0](https://physionet.org/content/nsrdb/view-license/1.0.0/)

## Supported Tasks

* [Rhythm](../tasks/rhythm.md)


## Usage

!!! Example Python

    ```py linenums="1"
    from pathlib import Path
    import heartkit as hk

    # Download dataset
    hk.datasets.download_datasets(hk.HKDownloadParams(
        datasets=[{
            "name": "mitbih",
            "params": {
                "path": "./datasets/mitbih"
            }
        }],
        progress=True
    ))
    ```
