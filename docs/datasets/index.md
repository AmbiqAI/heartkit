
# :material-database: Datasets

HeartKit provides support for a number of datasets to facilitate training the __heart-monitoring tasks__. Most of the datasets are readily available and can be downloaded and used for training and evaluation. The datasets inherit from [HKDataset](/heartkit/api/heartkit/datasets/dataset) and can be accessed either directly or through the factory singleton [`DatasetFactory`](#dataset-factory).

## <span class="sk-h2-span">Available Datasets</span>

Below is a list of the currently available datasets in HeartKit. Please make sure to review each dataset's license for terms and limitations.

* **[Icentia11k](./icentia11k.md)**: This dataset consists of ECG recordings from 11,000 patients and 2 billion labelled beats. The data was collected by the CardioSTAT, a single-lead heart monitor device from Icentia. The raw signals were recorded with a 16-bit resolution and sampled at 250 Hz with the CardioSTAT in a modified lead 1 position.

* **[LSAD](./lsad.md)**: The Large Scale Rhythm Database (LSAD) is a large publicly available electrocardiography dataset. It contains 10 second, 12-lead ECGs of 45,152 patients with a 500 Hz sampling rate. The ECGs are sampled at 500 Hz and are annotated by up to two cardiologists.

* **[LUDB](./ludb.md)**: Lobachevsky University Electrocardiography database consists of 200 10-second 12-lead records. The boundaries and peaks of P, T waves and QRS complexes were manually annotated by cardiologists. Each record is annotated with the corresponding diagnosis.

* **[PTB-XL](./ptbxl.md)**: The PTB-XL is a large publicly available electrocardiography dataset. It contains 21837 clinical 12-lead ECGs from 18885 patients of 10 second length. The ECGs are sampled at 500 Hz and are annotated by up to two cardiologists.

* **[QTDB](./qtdb.md)**: Over 100 fifteen-minute two-lead ECG recordings with onset, peak, and end markers for P, QRS, T, and (where present) U waves of from 30 to 50 selected beats in each recording.

* **[ECG Synthetic](./synthetic.md)**: An ECG synthetic dataset generated using PhysioKit. The dataset enables the generation of 12-lead ECG signals with a variety of heart conditions and noise levels along with segmentations and fiducial points.

* **[PPG Synthetic](./synthetic.md)**: A PPG synthetic dataset generated using PhysioKit. The dataset enables the generation of a 1-lead PPG signal with segmentations and fiducials.

* **[Bring-Your-Own-Data](./byod.md)**: Add new datasets to HeartKit by providing your own data. Subclass `HKDataset` and register it with the `DatasetFactory`.

## <span class="sk-h2-span">Dataset Factory</span>

The dataset factory, `DatasetFactory`, provides a convenient way to access the datasets. The factory is a thread-safe singleton class that provides a single point of access to the datasets via the datasets' slug names. The benefit of using the factory is it allows registering new additional datasets that can then be leveraged by existing and new tasks.

The dataset factory provides the following methods:

* **hk.DatasetFactory.register**: Register a custom dataset
* **hk.DatasetFactory.unregister**: Unregister a custom dataset
* **hk.DatasetFactory.has**: Check if a dataset is registered
* **hk.DatasetFactory.get**: Get a dataset
* **hk.DatasetFactory.list**: List all available datasets


```py linenums="1"

import heartkit as hk

# Grab EcgSynthetic dataset and instantiate it
Dataset = hk.DatasetFactory.get('ecg-synthetic')
ds = Dataset(num_pts=100)

# Grab the first patient's data and segmentations
with ds.patient_data(patient_id=ds.patient_ids[0]) as pt:
    ecg = pt["data"][:]
    segs = pt["segmentations"][:]

```
