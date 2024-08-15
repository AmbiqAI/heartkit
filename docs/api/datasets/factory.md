# Dataset Factory

See [Datasets](../../datasets/index.md) for information about available datasets.

To list all available dataset names and their corresponding classes:

## hk.DatasetFactory

```python
import heartkit as hk

for dataset in hk.DatasetFactory.list():
    print(f"Dataset name: {dataset} - {hk.DatasetFactory.get(dataset)}")
```

::: neuralspot_edge.utils.ItemFactory
