from ...defines import HKTaskParams
from ..task import HKTask
from . import datasets
from .datasets import FoundationTaskFactory
from .demo import demo
from .evaluate import evaluate
from .export import export
from .train import train


class FoundationTask(HKTask):
    """HeartKit Foundation Task"""

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
