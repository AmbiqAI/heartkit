# LUDB Dataset

## <span class="sk-h2-span">Overview</span>

The Lobachevsky University Electrocardiography database (LUDB) consists of 200 10-second 12-lead records. The boundaries and peaks of P, T waves and QRS complexes were manually annotated by cardiologists. Each record is annotated with the corresponding diagnosis.

Please visit [Physionet](https://physionet.org/content/ludb/1.0.1/) for more details.

## <span class="sk-h2-span">Usage</span>

!!! Example Python

    ```python
    from pathlib import Path
    import neuralspot_edge as nse
    import heartkit as hk

    ds = hk.DatasetFactory.get('ludb')(
        path=Path("./datasets/ludb")
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

The study was supported by the Ministry of Education of the Russian Federation (contract No. 02.G25.31.0157 of 01.12.2015).

## <span class="sk-h2-span">License</span>

The LUDB is available for commercial use.

## <span class="sk-h2-span">Supported Tasks</span>

* [Segmentation](../tasks/segmentation.md)
