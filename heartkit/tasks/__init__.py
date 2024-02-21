from .arrhythmia import (
    ArrhythmiaTask,
    HeartRate,
    HeartRateName,
    HeartRhythm,
    HeartRhythmName,
)
from .beat import BeatTask, HeartBeat, HeartBeatName
from .denoise import DenoiseTask
from .factory import TaskFactory
from .segmentation import HeartSegment, HeartSegmentName, SegmentationTask
from .task import HKTask

TaskFactory.register("arrhythmia", ArrhythmiaTask)
TaskFactory.register("beat", BeatTask)
TaskFactory.register("segmentation", SegmentationTask)
TaskFactory.register("denoise", DenoiseTask)
