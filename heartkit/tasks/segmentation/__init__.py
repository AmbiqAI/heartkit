"""
# Segmentation Task API

The objective of segmentation is to delineate key segments of the signal.
In the context of ECG signals, this involves identifying the different components of the cardiac cycle, including the P-wave, QRS complex, and T-wave.
These segments are used to compute a number of clinically relevant metrics, including heart rate, PR interval, QRS duration, QT interval, and QTc interval.
For PPG, the task involves segmenting the systolic and diastolic phases of the cardiac cycle.
Segmentation models are useful for detecting arrhythmias, heart rate variability, and other cardiac abnormalities.

Classes:
    SegmentationTask: Segmentation task

"""

from ...defines import HKTaskParams
from ..task import HKTask
from .defines import HKSegment
from .demo import demo
from .evaluate import evaluate
from .export import export
from .train import train


class SegmentationTask(HKTask):
    """heartKIT Segmentation Task"""

    @staticmethod
    def train(params: HKTaskParams):
        """Train model for segmentation task

        Args:
            params (HKTaskParams): Task parameters
        """
        train(params)

    @staticmethod
    def evaluate(params: HKTaskParams):
        """Evaluate segmentation model

        Args:
            params (HKTaskParams): Task parameters
        """
        evaluate(params)

    @staticmethod
    def export(params: HKTaskParams):
        """Export segmentation model

        Args:
            params (HKTaskParams): Task parameters
        """
        export(params)

    @staticmethod
    def demo(params: HKTaskParams):
        """Run segmentation demo.

        Args:
            params (HKTaskParams): Task parameters
        """
        demo(params)
