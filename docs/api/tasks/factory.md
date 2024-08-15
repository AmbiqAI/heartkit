# TaskFactory API

See [Tasks](../../tasks/index.md) for information about available tasks.

## hk.TaskFactory

```python
import heartkit as hk

for model in hk.TaskFactory.list():
    print(f"Task name: {model} - {hk.TaskFactory.get(model)}")
```

::: neuralspot_edge.utils.ItemFactory
