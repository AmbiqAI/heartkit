import abc

from ..defines import HKTaskParams


class HKTask(abc.ABC):
    """HeartKit Task base class. All tasks should inherit from this class."""

    @staticmethod
    def description() -> str:
        """Get task description

        Returns:
            str: description

        """
        return ""

    @staticmethod
    def train(params: HKTaskParams) -> None:
        """Train a model

        Args:
            params (HKTaskParams): train parameters

        """
        raise NotImplementedError

    @staticmethod
    def evaluate(params: HKTaskParams) -> None:
        """Evaluate a model

        Args:
            params (HKTaskParams): test parameters

        """
        raise NotImplementedError

    @staticmethod
    def export(params: HKTaskParams) -> None:
        """Export a model

        Args:
            params (HKTaskParams): export parameters

        """
        raise NotImplementedError

    @staticmethod
    def demo(params: HKTaskParams) -> None:
        """Run a demo

        Args:
            params (HKTaskParams): demo parameters

        """
        raise NotImplementedError
