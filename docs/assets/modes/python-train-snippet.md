```python
from pathlib import Path
import heartkit as hk

task = hk.TaskFactory.get("rhythm")
task.train(hk.HKTrainParams(
    job_dir=Path("./results/rhythm-class-2"),
    ds_path=Path("./datasets"),
    datasets=[{
        "name": "icentia11k",
        "params": {}
    }],
    num_classes=2,
    class_map={
        0: 0,
        1: 1,
        2: 1
    },
    class_names=[
        "NONE", "AFIB/AFL"
    ],
    sampling_rate=200,
    frame_size=800,
    samples_per_patient=[100, 800],
    val_samples_per_patient=[100, 800],
    data_parallelism=lambda: os.cpu_count() or 1,
    preprocesses=[
        hk.PreprocessParams(
            name="znorm",
            params=dict(
                eps=0.01,
                axis=None
            )
        )
    ]
))
```
