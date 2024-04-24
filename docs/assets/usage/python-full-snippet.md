```python
import heartkit as hk

ds_params = hk.HKDownloadParams.parse_file("download-datasets.json")
train_params = hk.HKTrainParams.parse_file("train.json")
test_params = hk.HKTestParams.parse_file("evaluate.json")
export_params = hk.HKExportParams.parse_file("export.json")

# Download datasets
hk.datasets.download_datasets(ds_params)

task = hk.TaskFactory.get("rhythm")

# Train rhythm model
task.train(train_params)

# Evaluate rhythm model
task.evaluate(test_params)

# Export rhythm model
task.export(export_params)

```
