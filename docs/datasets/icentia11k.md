# Icentia11k Dataset

## <span class="sk-h2-span">Overview</span>

This dataset consists of ECG recordings from 11,000 patients and 2 billion labelled beats. The data was collected by the CardioSTAT, a single-lead heart monitor device from Icentia. The raw signals were recorded with a 16-bit resolution and sampled at 250 Hz with the CardioSTAT in a modified lead 1 position. We provide derived version of the dataset where each patient is stored in separate [HDF5 files](https://www.hdfgroup.org/solutions/hdf5/) on S3. This makes it faster to download as well as makes it possible to leverage TensorFlow `prefetch` and `interleave` to parallelize data loading.

More info available on [PhysioNet website](https://physionet.org/content/icentia11k-continuous-ecg/1.0)

## <span class="sk-h2-span">Usage</span>

!!! Example Python

    ```python
    from pathlib import Path
    import neuralspot_edge as nse
    import heartkit as hk

    ds = hk.DatasetFactory.get('icentia11k')(
        path=Path("./datasets/icentia11k")
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

???+ note
    The __Icentia11k dataset__ requires roughly 200 GB of disk space and can take around 2 hours to download.


## <span class="sk-h2-span">Funding</span>

This work is partially funded by a grant from Icentia, Fonds de Recherche en Santé du Québec, and the Institute of Data Valorization (IVADO).

## <span class="sk-h2-span">Licensing</span>

The Icentia11k dataset is available for non-commercial use only.
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License](https://physionet.org/content/icentia11k-continuous-ecg/view-license/1.0/)

<!-- ## <span class="sk-h2-span">Supported Tasks</span>

* [Rhythm](../tasks/rhythm.md)
* [Beat](../tasks/beat.md)
* [2-Class Segmentation](../tasks/segmentation.md) -->

!!! warning
    The dataset is intended for evaluation purposes only and cannot be used for commercial use without permission. Please visit [Physionet](https://physionet.org/content/icentia11k-continuous-ecg/1.0) for more details.
