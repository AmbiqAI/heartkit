"""
# Denoise Task API

The objective of denoising is to remove noise and artifacts from physiological signals while preserving the underlying signal information.
The dominant noise sources include baseline wander (BW), muscle noise (EMG), electrode movement artifacts (EM), and powerline interference (PLI).
For physiological signals such as ECG and PPG, removing the artifacts is difficult due to the non-stationary nature of the noise and overlapping frequency bands with the signal.
While traditional signal processing techniques such as filtering and wavelet denoising have been used to remove noise, deep learning models have shown great promise in enhanced denoising.

Classes:
    DenoiseTask: Denoising task

"""

from ...defines import HKTaskParams
from ..task import HKTask
from .demo import demo
from .evaluate import evaluate
from .export import export
from .train import train
from .dataloaders import DenoiseTaskFactory


class DenoiseTask(HKTask):
    """heartKIT Denoise Task"""

    @staticmethod
    def train(params: HKTaskParams):
        """Train model for denoise task

        Args:
            params (HKTaskParams): Task parameters
        """
        train(params)

    @staticmethod
    def evaluate(params: HKTaskParams):
        """Evaluate denoise model

        Args:
            params (HKTaskParams): Task parameters
        """
        evaluate(params)

    @staticmethod
    def export(params: HKTaskParams):
        """Export denoise model

        Args:
            params (HKTaskParams): Task parameters
        """
        export(params)

    @staticmethod
    def demo(params: HKTaskParams):
        """Run denoise demo.

        Args:
            params (HKTaskParams): Task parameters
        """
        demo(params)
