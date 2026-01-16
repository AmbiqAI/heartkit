"""
# Foundation Task API

The objective of this task is to create a foundation model that can be used to build downstream models for other tasks.
In heartKIT, the foundation model is trained using SimCLR- using two augmented leads as input.

Classes:
    FoundationTask: Foundation task

"""

from ...defines import HKTaskParams
from ..task import HKTask
from . import datasets
from .datasets import FoundationTaskFactory
from .demo import demo
from .evaluate import evaluate
from .export import export
from .train import train


class FoundationTask(HKTask):
    """heartKIT Foundation Task"""

    @staticmethod
    def train(params: HKTaskParams):
        """Train model for foundation task using SimCLR

        Args:
            params (HKTaskParams): Task parameters
        """
        train(params)

    @staticmethod
    def evaluate(params: HKTaskParams):
        """Evaluate model for foundation task using SimCLR

        Args:
            params (HKTaskParams): Task parameters
        """
        evaluate(params)

    @staticmethod
    def export(params: HKTaskParams):
        """Export foundation task model

        Args:
            params (HKTaskParams): Task parameters
        """
        export(params)

    @staticmethod
    def demo(params: HKTaskParams):
        """Run demo for foundation task model

        Args:
            params (HKTaskParams): Task parameters
        """
        demo(params)
