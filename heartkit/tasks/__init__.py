from .beat import BeatTask, HKBeat
from .denoise import DenoiseTask
from .diagnostic import DiagnosticTask, HKDiagnostic
from .factory import TaskFactory
from .foundation import FoundationTask
from .rhythm import HeartRate, HKRhythm, RhythmTask
from .segmentation import HKSegment, SegmentationTask
from .task import HKTask

TaskFactory.register("rhythm", RhythmTask)
TaskFactory.register("beat", BeatTask)
TaskFactory.register("segmentation", SegmentationTask)
TaskFactory.register("diagnostic", DiagnosticTask)
TaskFactory.register("denoise", DenoiseTask)
TaskFactory.register("foundation", FoundationTask)
