"""
# :simple-task: Tasks API

heartKIT provides several built-in __heart-monitoring__ tasks.
Each task is designed to address a unique aspect such as ECG denoising, segmentation, and rhythm/beat classification.
The tasks are designed to be modular and can be used independently or in combination to address specific use cases.

## Available Tasks

- **[BeatTask](./beat)**: Beat classification task
- **[DenoiseTask](./denoise)**: Denoising task
- **[FoundationTask](./foundation)**: Foundation task
- **[RhythmTask](./rhythm)**: Rhythm classification task
- **[SegmentationTask](./segmentation)**: Segmentation task

## Task Factory

The TaskFactory provides a convenient way to access the built-in tasks.
The factory is a thread-safe singleton class that provides a single point of access to the tasks via the tasks' slug names.
The benefit of using the factory is it allows registering custom tasks that can then be used just like built-in tasks.

```python
import heartkit as hk

for task in hk.TaskFactory.list():
    print(f"Task name: {task} - {hk.TaskFactory.get(task)}")
```

Classes:
    HKTask: Base class for all tasks
    BeatTask: Beat classification task
    HKBeat: Beat task class labels
    DenoiseTask: Denoising task
    DiagnosticTask: Diagnostic task
    HKDiagnostic: Diagnostic task class labels
    FoundationTask: Foundation task
    HKRhythm: Rhythm task class labels
    RhythmTask: Rhythm classification task
    HKSegment: Segmentation task class labels
    SegmentationTask: Segmentation task

"""

import helia_edge as helia

from . import beat, denoise, diagnostic, foundation, rhythm, segmentation, utils

from .beat import BeatTask, HKBeat
from .denoise import DenoiseTask
from .diagnostic import DiagnosticTask, HKDiagnostic
from .foundation import FoundationTask
from .rhythm import HKRhythm, RhythmTask
from .segmentation import HKSegment, SegmentationTask
from .task import HKTask
from .translate import HKTranslate, TranslateTask

TaskFactory = helia.utils.create_factory(factory="HKTaskFactory", type=HKTask)

TaskFactory.register("rhythm", RhythmTask)
TaskFactory.register("beat", BeatTask)
TaskFactory.register("segmentation", SegmentationTask)
TaskFactory.register("diagnostic", DiagnosticTask)
TaskFactory.register("denoise", DenoiseTask)
TaskFactory.register("foundation", FoundationTask)
TaskFactory.register("translate", TranslateTask)
