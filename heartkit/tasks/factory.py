from .arrhythmia import Arrhythmia
from .beat import Beat
from .denoise import Denoise
from .segmentation import Segmentation
from .task import HKTask

_tasks: dict[str, HKTask] = {"arrhythmia": Arrhythmia, "beat": Beat, "segmentation": Segmentation, "denoise": Denoise}


class TaskFactory:
    """Task factory enables registering, creating, and listing tasks. It is a singleton class."""

    @staticmethod
    def register(name: str, task: HKTask) -> None:
        """Register a task

        Args:
            name (str): task name
            task (type[HKTask]): task
        """
        _tasks[name] = task

    @staticmethod
    def unregister(name: str) -> None:
        """Unregister a task

        Args:
            name (str): task name
        """
        del _tasks[name]

    @staticmethod
    def create(name: str, **kwargs) -> HKTask:
        """Create a task

        Args:
            name (str): task name

        Returns:
            HKTask: task
        """
        return _tasks[name](**kwargs)

    @staticmethod
    def list() -> list[str]:
        """List registered tasks

        Returns:
            list[str]: task names
        """
        return list(_tasks.keys())

    @staticmethod
    def get(name: str) -> HKTask:
        """Get a task

        Args:
            name (str): task name

        Returns:
            HKTask: task
        """
        return _tasks[name]

    @staticmethod
    def has(name: str) -> bool:
        """Check if a task is registered

        Args:
            name (str): task name

        Returns:
            bool: True if registered
        """
        return name in _tasks
