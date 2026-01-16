"""
# Translate Task API

The objective of translation is to convert signals from one physiological modality to another.
For example, translating ECG signals to PPG signals or vice versa.

Classes:
    TranslateTask: Translate task

"""

from ...defines import HKTaskParams
from ..task import HKTask
from .defines import HKTranslate
from .demo import demo
from .evaluate import evaluate
from .export import export
from .train import train


class TranslateTask(HKTask):
    """heartKIT Translate Task"""

    @staticmethod
    def train(params: HKTaskParams):
        """Train translate task model with given parameters.

        Args:
            params (HKTaskParams): Task parameters
        """
        train(params)

    @staticmethod
    def evaluate(params: HKTaskParams):
        """Evaluate translation task model with given parameters.

        Args:
            params (HKTaskParams): Task parameters
        """
        evaluate(params)

    @staticmethod
    def export(params: HKTaskParams):
        """Export translation task model with given parameters.

        Args:
            params (HKTaskParams): Task parameters
        """
        export(params)

    @staticmethod
    def demo(params: HKTaskParams):
        """Run translate demo for model

        Args:
            params (HKTaskParams): Task parameters
        """
        demo(params)
