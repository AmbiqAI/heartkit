# QTDB Dataset

## <span class="sk-h2-span">Overview</span>

Over 100 fifteen-minute two-lead ECG recordings with onset, peak, and end markers for P, QRS, T, and (where present) U waves of from 30 to 50 selected beats in each recording.

Please visit [Physionet](https://doi.org/10.13026/C24K53) for more details.

!!! Example Python

    ```python
    from pathlib import Path
    import neuralspot_edge as nse
    import heartkit as hk

    ds = hk.DatasetFactory.get('qtdb')(
        path=Path("./datasets/qtdb")
    )

    # Download dataset
    ds.download(force=False)

    # Create signal generator
    data_gen = self.ds.signal_generator(
        patient_generator=nse.utils.uniform_id_generator(ds.patient_ids, repeat=True, shuffle=True),
        frame_size=256,
        samples_per_patient=5,
        target_rate=100,
    )

    # Grab single ECG sample
    ecg = next(data_gen)

    ```

## <span class="sk-h2-span">Funding</span>

The QT Database was created as part of a project funded by the National Library of Medicine.

## <span class="sk-h2-span">License</span>

The QT Database is available for commercial use. [Open Data Commons Attribution License v1.0](https://physionet.org/content/qtdb/view-license/1.0.0/)

## <span class="sk-h2-span">Supported Tasks</span>

* [Segmentation](../tasks/segmentation.md)
