from ...defines import HKTaskParams
from ..task import HKTask
from .defines import HKBeat
from .demo import demo
from .evaluate import evaluate
from .export import export
from .train import train


class BeatTask(HKTask):
    """HeartKit Beat Task"""

    @staticmethod
    def description() -> str:
        return (
            "This task is used to train, evaluate, and export beat models."
            "Beat includes normal, pac, pvc, and other beats."
        )

    @staticmethod
    def train(params: HKTaskParams):
        train(params)

    @staticmethod
    def evaluate(params: HKTaskParams):
        evaluate(params)

    @staticmethod
    def export(params: HKTaskParams):
        export(params)

    @staticmethod
    def demo(params: HKTaskParams):
        demo(params)
