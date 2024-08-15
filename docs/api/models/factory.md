# ModelFactory API

See [Models](../../models/index.md) for information about available models.

## hk.ModelFactory

```python
import heartkit as hk

for model in hk.ModelFactory.list():
    print(f"Model name: {model} - {hk.ModelFactory.get(model)}")
```
