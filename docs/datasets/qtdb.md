# QTDB Dataset

## Overview

Over 100 fifteen-minute two-lead ECG recordings with onset, peak, and end markers for P, QRS, T, and (where present) U waves of from 30 to 50 selected beats in each recording.

Please visit [Physionet](https://doi.org/10.13026/C24K53) for more details.

!!! Example Python

    ```py linenums="1"
    from pathlib import Path
    import helia_edge as helia
    import heartkit as hk

    ds = hk.DatasetFactory.get('qtdb')(
        path=Path("./datasets/qtdb")
    )

    # Download dataset
    ds.download(force=False)

    # Create signal generator
    data_gen = self.ds.signal_generator(
        patient_generator=helia.utils.uniform_id_generator(ds.patient_ids, repeat=True, shuffle=True),
        frame_size=256,
        samples_per_patient=5,
        target_rate=100,
    )

    # Grab single ECG sample
    ecg = next(data_gen)

    ```

## Funding

The QT Database was created as part of a project funded by the National Library of Medicine.

## License

The QT Database is available for commercial use. [Open Data Commons Attribution License v1.0](https://physionet.org/content/qtdb/view-license/1.0.0/)

## Supported Tasks

* [Segmentation](../tasks/segmentation.md)
