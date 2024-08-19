# Python Usage

__HeartKit__ python package allows for more fine-grained control and customization. You can use the package to train, evaluate, and deploy models for both built-in taks and custom tasks. In addition, custom datasets and model architectures can be created and registered with corresponding factories.

## <span class="sk-h2-span">Overview</span>

The main components of HeartKit include the following:

### [Tasks](../tasks/index.md)

A [Task](../tasks/index.md) inherits from the [HKTask](../api/tasks/task.md#hkhktask) class and provides implementations for each of the main modes: download, train, evaluate, export, and demo. Each mode is provided with a set of parameters defined by [HKTaskParams](../api/tasks/task.md#hkhktaskparams). Additional task-specific parameters can be extended to the `HKTaskParams` class. These tasks are then registered and accessed via the [TaskFactory](../api/tasks/factory.md) using a unique task name as the key and the custom Task class as the value.

```python
import heartkit as hk

task = hk.TaskFactory.get('rhythm')
```

### [Datasets](../datasets/index.md)

A dataset inherits from the [HKDataset](../api/datasets/dataset.md#hkdatasethkdataset) class and provides implementations for downloading, preparing, and loading the dataset. Each dataset is provided with a set of custom parameters for initialization. Since each task will require specific transformations of the data, the dataset class provides only a general interface for loading the data. Each task must then provide a set of corresponding [HKDataloader](../api/datasets/dataloader.md) classes to transform the dataset into a format that can be used by the task. The datasets are registered and accessed via the [DatasetFactory](../api/datasets/factory.md) using a unique dataset name as the key and the Dataset class as the value. Each Task can then create its own `DataloaderFactory` that will provide a corresponding dataloader for each supported dataset. The Task's `DataloaderFactory` should use the same dataset names as the DatasetFactory to ensure that the correct dataloader is used for each dataset.

```python
import heartkit as hk

ds = hk.DatasetFactory.get('ecg-synthetic')(num_pts=100)
```

### [Models](../models/index.md)

Lastly, HeartKit leverages [neuralspot-edge's](https://ambiqai.github.io/neuralspot-edge/) customizable model architectures. To enable creating custom network topologies from configuration files, HeartKit provides a [ModelFactory](../api/models/factory.md) that allows you to create models by specifying the model key and the model parameters. Each item in the factory is a callable that takes a `keras.Input`, model parameters, and number of classes as arguments and returns a `keras.Model`.

```
import keras
import heartkit as hk

inputs = keras.Input((256, 1), dtype="float32")
num_classes = 4
model_params = dict(...)

model = hk.ModelFactory.get('tcn')(
    x=inputs,
    params=model_params,
    num_classes=num_classes
)

```

## <span class="sk-h2-span">Usage</span>

### Running a built-in task w/ existing datasets

1. Create a task configuration file defining the model, datasets, class labels, mode parameters, and so on. Have a look at the [HKTaskParams](../modes/configuration.md#hktaskparams) for more details on the available parameters.

2. Leverage [TaskFactory](../api/tasks/factory.md) to get the desired built-in task.

3. Run the task's main modes: `download`, `train`, `evaluate`, `export`, and/or `demo`.


```python

import heartkit as hk

params = hk.HKTaskParams(...)  # (1)

task = hk.TaskFactory.get("rhythm")

task.download(params)  # Download dataset(s)

task.train(params)  # Train the model

task.evaluate(params)  # Evaluate the model

task.export(params)  # Export to TFLite

```

1. Example configuration:
--8<-- "assets/usage/python-configuration.md"

### Running a custom task w/ custom datasets

To create a custom task, check out the [Bring-Your-Own-Task Guide](../tasks/byot.md).

To create a custom dataset, check out the [Bring-Your-Own-Dataset Guide](../datasets/byod.md).

---
