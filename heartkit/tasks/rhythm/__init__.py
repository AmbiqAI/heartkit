"""
# Rhythm Task API

The objective of rhythm classification is to detect and classify abnormal heart rhythms, also known as arrhythmias, directly from ECG signals.

Classes:
    RhythmTask: Rhythm classification task

"""

from ...defines import HKTaskParams
from ..task import HKTask
from .defines import HKRhythm
from .demo import demo
from .evaluate import evaluate
from .export import export
from .train import train


class RhythmTask(HKTask):
    """heartKIT Rhythm Task"""

    @staticmethod
    def description() -> str:
        return (
            "This task is used to train, evaluate, and export rhythm models."
            "Rhythm includes sinus rhythm, atrial fibrillation, and other arrhythmias."
        )

    @staticmethod
    def train(params: HKTaskParams):
        """Train model for rhythm task

        Args:
            params (HKTaskParams): Task parameters
        """
        train(params)

    @staticmethod
    def evaluate(params: HKTaskParams):
        """Evaluate rhythm task model

        Args:
            params (HKTaskParams): Task parameters
        """
        evaluate(params)

    @staticmethod
    def export(params: HKTaskParams):
        """Export model for rhythm task

        Args:
            params (HKTaskParams): Task parameters
        """
        export(params)

    @staticmethod
    def demo(params: HKTaskParams):
        """Run demo on rhythm task model

        Args:
            params (HKTaskParams): Task parameters
        """
        demo(params)
