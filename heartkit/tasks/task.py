import abc

from ..defines import HKDemoParams, HKExportParams, HKTestParams, HKTrainParams


class HKTask(abc.ABC):
    """HeartKit Task base class. All tasks should inherit from this class."""

    @staticmethod
    def train(params: HKTrainParams) -> None:
        """Train a model

        Args:
            params (HKTrainParams): train parameters

        """
        raise NotImplementedError

    @staticmethod
    def evaluate(params: HKTestParams) -> None:
        """Evaluate a model

        Args:
            params (HKTestParams): test parameters

        """
        raise NotImplementedError

    @staticmethod
    def export(params: HKExportParams) -> None:
        """Export a model

        Args:
            params (HKExportParams): export parameters

        """
        raise NotImplementedError

    @staticmethod
    def demo(params: HKDemoParams) -> None:
        """Run a demo

        Args:
            params (HKDemoParams): demo parameters

        """
        raise NotImplementedError
