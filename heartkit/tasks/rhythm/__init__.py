from ...defines import HKDemoParams, HKExportParams, HKTestParams, HKTrainParams
from ..task import HKTask
from .defines import HKRhythm
from .demo import demo
from .evaluate import evaluate
from .export import export
from .train import train


class RhythmTask(HKTask):
    """HeartKit Rhythm Task"""

    @staticmethod
    def description() -> str:
        return (
            "This task is used to train, evaluate, and export rhythm models."
            "Rhythm includes sinus rhythm, atrial fibrillation, and other arrhythmias."
        )

    @staticmethod
    def train(params: HKTrainParams):
        train(params)

    @staticmethod
    def evaluate(params: HKTestParams):
        evaluate(params)

    @staticmethod
    def export(params: HKExportParams):
        export(params)

    @staticmethod
    def demo(params: HKDemoParams):
        demo(params)
