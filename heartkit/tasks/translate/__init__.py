from ...defines import HKTaskParams
from ..task import HKTask
from .defines import HKTranslate
from .demo import demo
from .evaluate import evaluate
from .export import export
from .train import train


class TranslateTask(HKTask):
    """HeartKit Translate Task"""

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
