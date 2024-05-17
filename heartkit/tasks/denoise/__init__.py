from ...defines import HKDemoParams, HKExportParams, HKTestParams, HKTrainParams
from ..task import HKTask
from .demo import demo
from .evaluate import evaluate
from .export import export
from .train import train


class DenoiseTask(HKTask):
    """HeartKit Denoise Task"""

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
