# Large Scale Arrhythmia Database (LSAD)

## <span class="sk-h2-span">Overview</span>

The large scale arrhythmia database (LSAD) is a large-scale, multi-center, multi-lead, and multi-class ECG dataset for arrhythmia detection. The dataset contains ECG recordings from 45,152 patients. The dataset is collected from 3 different centers: Shaoxing People's Hospital, the Second Affiliated Hospital of Zhejiang University, and the First Affiliated Hospital of Zhejiang University. The dataset contains 11 different arrhythmia classes and 1 normal class. The dataset is collected from 12-lead ECGs and is annotated by a team of expert cardiologists. The dataset includes over 100 scp codes.

Please visit [Physionet](https://physionet.org/content/ecg-arrhythmia/1.0.0/) for more details.

## <span class="sk-h2-span">Usage</span>

!!! Example Python

    ```python
    from pathlib import Path
    import neuralspot_edge as nse
    import heartkit as hk

    ds = hk.DatasetFactory.get('lsad')(
        path=Path("./datasets/lsad")
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

## <span class="sk-h2-span">Statistics</span>

| Acronym Name | Full Name | Frequency, n(%) | Age, Mean ± SD |Male,n(%) |
| --- | --- | --- | --- | --- |
| SB | Sinus Bradycardia | 15,528 (38.6) | 58.4 ± 14.02 | 9844 (63.4%) |
| SR | Sinus Rhythm | 7,291 (18.1) | 54.38 ± 16.17 | 4107 (56.33%) |
| AFIB | Atrial Fibrillation | 7,028 (17.5) | 73.07 ± 11.27 | 4051 (57.64%) |
| ST | Sinus Tachycardia | 6,208 (15.4) | 54.24 ± 21.41 | 3208 (51.68%) |
| AFL | Atrial Flutter | 1,725 (4.3) | 71.57 ± 13.23 | 1001 (58.03%) |
| SI | Sinus Irregularity | 1,773 (4.4) | 37.3 ± 22.98 | 979 (55.22%) |
| SVT | Supraventricular Tachycardia | 542 (1.3) | 55.44 ± 18.41 | 289 (53.32%) |
| AT | Atrial Tachycardia | 133 (0.3) | 65.92 ± 18.7 | 69 (51.88%) |
| AVNRT | Atrioventricular Node Reentrant Tachycardia | 16 (0.03) | 57.88 ± 17.34 | 12 (75%) |
| AVRT | Atrioventricular Reentrant Tachycardia | 7 (0.01) | 56.43 ± 17.89 | 5 (71.43%) |
| WAP | Wandering Atrial Pacemaker | 7 (0.01) | 51.14 ± 31.83 | 6 (85.71%) |

## <span class="sk-h2-span">Funding</span>

This dataset received funding from the Kay Family Foundation Data Analytic Grant. This dataset received funding from 2018 Shaoxing Medical and Hygiene Research Grant, ID 2018C30070.

## <span class="sk-h2-span">License</span>

The dataset is available under [Creative Commons Attribution 4.0 International Public License](https://physionet.org/content/ecg-arrhythmia/view-license/1.0.0/)
