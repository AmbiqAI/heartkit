```python
from pathlib import Path
import heartkit as hk

task = hk.TaskFactory.get("segmentation")
task.export(hk.HKDemoParams(
    job_dir=Path("./results/segmentation-class-2"),
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
    sampling_rate=100,
    frame_size=256,
    backend="pc",
    model_file=Path("./results/segmentation-class-2/model.keras"),
))
```
