# PTB-XL Diagnostics Dataset

### <span class="sk-h2-span">Overview</span>

This dataset consists of 21837 clinical 12-lead ECGs from 18885 patients. The ECGs were recorded at a sampling frequency of 500 Hz and a resolution of 16 bits per sample. The dataset includes 21837 ECGs, 21837 ECGs with diagnostic labels, and 21837 ECGs with rhythm labels. The diagnostic labels include 71 classes, and the rhythm labels include 9 classes.

Please visit [Physionet](https://physionet.org/content/ptb-xl/1.0.3/) for more details.

### <span class="sk-h2-span">Funding</span>

This work was supported by BMBF (01IS14013A), Berlin Big Data Center, Berlin Center for Machine Learning, and EMPIR project 18HLT07 MedalCare.

### <span class="sk-h2-span">License</span>

This database is available under [Creative Commons Attribution 4.0 International Public License](https://physionet.org/content/ptb-xl/view-license/1.0.3/)

### <span class="sk-h2-span">Supported Tasks</span>

* [Rhythm](../tasks/rhythm.md)

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

## <span class="sk-h2-span">References</span>

* [Deep Learning for ECG Analysis: Benchmarks and Insights from PTB-XL](https://arxiv.org/pdf/2004.13701.pdf)
