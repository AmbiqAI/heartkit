# Synthetic Datasets

### <span class="sk-h2-span">Overview</span>

By leveraging [PhysioKit](https://ambiqai.github.io/physiokit/), we are able to generate synthetic data for a variety of physiological signals, including ECG, PPG, and respiration. In addition to the signals, the tool also provides corresponding landmark fiducials and segmentation annotations. While not a replacement for real-world data, synthetic data can be useful in conjunction with real-world data for training and testing the models.

## <span class="sk-h2-span">Available Datasets</span>

### ECG Synthetic

An ECG synthetic dataset generated using PhysioKit. The dataset enables the generation of 12-lead ECG signals with a variety of heart conditions and noise levels along with segmentations and fiducial points.

### PPG Synthetic

A PPG synthetic dataset generated using PhysioKit. The dataset enables the generation of a 1-lead PPG signal with segmentations and fiducials.

## <span class="sk-h2-span">Usage</span>

!!! Example Python

    ```py linenums="1"
    import heartkit as hk

    ds = hk.DatasetFactory.get('ecg-synthetic')(
        num_pts=100,
        params=dict(
            sample_rate=1000, # Hz
            duration=10, # seconds
            heart_rate=(40, 120),
        )
    )

    with ds.patient_data(patient_id=ds.patient_ids[0]) as pt:
        ecg = pt["data"][:]
        segs = pt["segmentations"][:]
        fids = pt["fiducials"][:]

    ```

    <div class="sk-plotly-graph-div">
    --8<-- "assets/tasks/segmentation/segmentation-example.html"
    </div>


## <span class="sk-h2-span">Funding</span>

NA

## <span class="sk-h2-span">Licensing</span>

The tool is available under BSD-3-Clause Licehelia.
