```python
from pathlib import Path
import heartkit as hk

task = hk.TaskFactory.get("rhythm")

task.demo(hk.HKDemoParams(
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
    model_file=Path("./results/rhythm-class-2/model.tflite"),
    backend="pc"
))
```
