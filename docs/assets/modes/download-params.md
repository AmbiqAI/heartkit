| Argument | Type | Opt/Req | Default | Description |
| --- | --- | --- | --- | --- |
| ds_path | Path | Optional | `Path()` | Dataset root directory |
| datasets | list[DatasetTypes] | Optional |  | Datasets |
| progress | bool | Optional | True | Display progress bar |
| force | bool | Optional | False | Force download dataset- overriding existing files |
| data_parallelism | int | Optional | `lambda: os.cpu_count() or 1` | # of data loaders running in parallel |
