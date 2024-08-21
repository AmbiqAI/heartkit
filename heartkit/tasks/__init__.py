import neuralspot_edge as nse

from . import beat, denoise, diagnostic, foundation, rhythm, segmentation, utils

from .beat import BeatTask, HKBeat
from .denoise import DenoiseTask
from .diagnostic import DiagnosticTask, HKDiagnostic
from .foundation import FoundationTask
from .rhythm import HKRhythm, RhythmTask
from .segmentation import HKSegment, SegmentationTask
from .task import HKTask
from .translate import HKTranslate, TranslateTask

TaskFactory = nse.utils.create_factory(factory="HKTaskFactory", type=HKTask)

TaskFactory.register("rhythm", RhythmTask)
TaskFactory.register("beat", BeatTask)
TaskFactory.register("segmentation", SegmentationTask)
TaskFactory.register("diagnostic", DiagnosticTask)
TaskFactory.register("denoise", DenoiseTask)
TaskFactory.register("foundation", FoundationTask)
TaskFactory.register("translate", TranslateTask)
