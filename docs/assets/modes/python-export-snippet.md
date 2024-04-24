```python
from pathlib import Path
import heartkit as hk

task = hk.TaskFactory.get("rhythm")
task.export(hk.HKExportParams(
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
    test_samples_per_patient=[100, 500, 100],
    model_file=Path("./results/rhythm-class-2/model.keras"),
    quantization={
        enabled=True,
        qat=False,
        ptq=True,
        input_type="int8",
        output_type="int8",
    },
    threshold=0.95,
    tflm_var_name="g_arrhythmia_model",
    tflm_file=Path("./results/rhythm-class-2/arrhythmia_model_buffer.h")
))
```
