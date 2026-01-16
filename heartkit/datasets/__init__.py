"""
# :material-database: Datasets API

heartKIT provides a number of datasets that can be used for training and evaluation of __heart-monitoring tasks__.

## Available Datasets

* **[Icentia11k](./icentia11k)**: This dataset consists of ECG recordings from 11,000 patients and 2 billion labelled beats. The data was collected by the CardioSTAT, a single-lead heart monitor device from Icentia. The raw signals were recorded with a 16-bit resolution and sampled at 250 Hz with the CardioSTAT in a modified lead 1 position.
* **[LSAD](./lsad)**: The Large Scale Rhythm Database (LSAD) is a large publicly available electrocardiography dataset. It contains 10 second, 12-lead ECGs of 45,152 patients with a 500 Hz sampling rate. The ECGs are sampled at 500 Hz and are annotated by up to two cardiologists.
* **[LUDB](./ludb)**: Lobachevsky University Electrocardiography database consists of 200 10-second 12-lead records. The boundaries and peaks of P, T waves and QRS complexes were manually annotated by cardiologists. Each record is annotated with the corresponding diagnosis.
* **[PTB-XL](./ptbxl)**: The PTB-XL is a large publicly available electrocardiography dataset. It contains 21837 clinical 12-lead ECGs from 18885 patients of 10 second length. The ECGs are sampled at 500 Hz and are annotated by up to two cardiologists.
* **[QTDB](./qtdb)**: Over 100 fifteen-minute two-lead ECG recordings with onset, peak, and end markers for P, QRS, T, and (where present) U waves of from 30 to 50 selected beats in each recording.
* **[ECG Synthetic](./ppg_synthetic)**: An ECG synthetic dataset generated using physioKIT. The dataset enables the generation of 12-lead ECG signals with a variety of heart conditions and noise levels along with segmentations and fiducial points.
* **[PPG Synthetic](./ecg_synthetic)**: A PPG synthetic dataset generated using physioKIT. The dataset enables the generation of a 1-lead PPG signal with segmentations and fiducials.

## Dataset Factory

The dataset factory, `DatasetFactory`, provides a convenient way to access the datasets.
The factory is a thread-safe singleton class that provides a single point of access to the datasets via the datasets' slug names.
The benefit of using the factory is it allows registering new additional datasets that can then be leveraged by existing and new tasks.

## Usage

```py linenums="1"

import heartkit as hk

# Grab EcgSynthetic dataset and instantiate it
ds = hk.DatasetFactory.get('ecg-synthetic')(
    num_pts=100
)

# Grab the first patient's data and segmentations
with ds.patient_data(patient_id=ds.patient_ids[0]) as pt:
    ecg = pt["data"][:]
    segs = pt["segmentations"][:]

```

Classes:
    IcentiaDataset: Icentia11k dataset
    LsadDataset: LSAD dataset
    LudbDataset: LUDB dataset
    PtbxlDataset: PTB-XL dataset
    QtdbDataset: QTDB dataset
    EcgSyntheticDataset: ECG synthetic dataset
    PpgSyntheticDataset: PPG synthetic

"""

from .augmentation import create_augmentation_pipeline
from .bidmc import BidmcDataset
from .dataset import HKDataset
from .defines import PatientGenerator
from .dataloader import HKDataloader
from .icentia11k import IcentiaDataset, IcentiaBeat, IcentiaRhythm
from .icentia_mini import IcentiaMiniDataset, IcentiaMiniRhythm, IcentiaMiniBeat
from .lsad import LsadDataset, LsadScpCode
from .ludb import LudbDataset, LudbSegmentation
from .nstdb import NstdbNoise
from .ptbxl import PtbxlDataset, PtbxlScpCode
from .qtdb import QtdbDataset
from .ecg_synthetic import EcgSyntheticDataset
from .ppg_synthetic import PpgSyntheticDataset
from .factory import DatasetFactory

DatasetFactory.register("bidmc", BidmcDataset)
DatasetFactory.register("ecg-synthetic", EcgSyntheticDataset)
DatasetFactory.register("ppg-synthetic", PpgSyntheticDataset)
DatasetFactory.register("icentia11k", IcentiaDataset)
DatasetFactory.register("icentia_mini", IcentiaMiniDataset)
DatasetFactory.register("lsad", LsadDataset)
DatasetFactory.register("ludb", LudbDataset)
DatasetFactory.register("qtdb", QtdbDataset)
DatasetFactory.register("ptbxl", PtbxlDataset)
