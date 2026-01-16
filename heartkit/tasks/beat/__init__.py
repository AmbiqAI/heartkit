"""
# Beat Task API

In beat classification, we classify individual beats as either normal or abnormal.
Abnormal beats can be further classified as being either premature or escape beats as well as originating from the atria, junction, or ventricles.
The objective of beat classification is to detect and classify these abnormal heart beats directly from ECG signals.

Classes:
    BeatTask: Beat classification task

"""

from ...defines import HKTaskParams
from ..task import HKTask
from .defines import HKBeat
from .demo import demo
from .evaluate import evaluate
from .export import export
from .train import train


class BeatTask(HKTask):
    """heartKIT Beat Task"""

    @staticmethod
    def description() -> str:
        return (
            "This task is used to train, evaluate, and export beat models."
            "Beat includes normal, pac, pvc, and other beats."
        )

    @staticmethod
    def train(params: HKTaskParams):
        """Train model for beat task

        Args:
            params (HKTaskParams): Task parameters
        """
        train(params)

    @staticmethod
    def evaluate(params: HKTaskParams):
        """Evaluate beat task model

        Args:
            params (HKTaskParams): Task parameters
        """
        evaluate(params)

    @staticmethod
    def export(params: HKTaskParams):
        """Export model for beat task

        Args:
            params (HKTaskParams): Task parameters
        """
        export(params)

    @staticmethod
    def demo(params: HKTaskParams):
        """Run demo on beat task model

        Args:
            params (HKTaskParams): Task parameters
        """
        demo(params)
