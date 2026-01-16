"""
# Diagnostic Task API

The objective of diagnostic classification is to detect and classify abnormal heart conditions directly from ECG signals.

Classes:
    DiagnosticTask: Diagnostic classification task

"""

from ...defines import HKTaskParams
from ..task import HKTask
from .defines import HKDiagnostic
from .demo import demo
from .evaluate import evaluate
from .export import export
from .train import train


class DiagnosticTask(HKTask):
    """heartKIT Diagnostic Task"""

    @staticmethod
    def train(params: HKTaskParams):
        """Train diagnostic task model with given parameters.

        Args:
            params (HKTaskParams): Task parameters
        """
        train(params)

    @staticmethod
    def evaluate(params: HKTaskParams):
        """Evaluate diagnostic task model with given parameters.

        Args:
            params (HKTaskParams): Task parameters
        """
        evaluate(params)

    @staticmethod
    def export(params: HKTaskParams):
        """Export model

        Args:
            params (HKTaskParams): Task parameters
        """
        export(params)

    @staticmethod
    def demo(params: HKTaskParams):
        """Run diagnostic demo.

        Args:
            params (HKTaskParams): Task parameters
        """
        demo(params)
