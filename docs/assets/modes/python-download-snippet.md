```python
from pathlib import Path
import heartkit as hk

hk.datasets.download_datasets(hk.HKDownloadParams(
    ds_path=Path("./datasets"),
    datasets=["icentia11k", "ludb", "qtdb", "ecg-synthetic"],
    progress=True
))
```
