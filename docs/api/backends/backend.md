# Inference Engine Backend API

## hk.HKInferenceBackend

::: heartkit.backends.HKInferenceBackend


## hk.BackendFactory

```python
import heartkit as hk

for backend in hk.BackendFactory.list():
    print(f"Backend name: {backend} - {hk.BackendFactory.get(backend)}")
```

::: neuralspot_edge.utils.ItemFactory
