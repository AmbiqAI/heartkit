from typing import TypeVar, Generic, Type
from threading import Lock

T = TypeVar("T")


class SingletonMeta(type):
    """Thread-safe singleton."""

    _instances = {}
    _lock: Lock = Lock()

    def __call__(cls, *args, **kwargs):
        with cls._lock:
            if "singleton" in kwargs:
                instance_name = kwargs.get("singleton")
                del kwargs["singleton"]
            else:
                instance_name = cls
            # END IF
            if instance_name not in cls._instances:
                instance = super().__call__(*args, **kwargs)
                cls._instances[instance_name] = instance
        return cls._instances[instance_name]


class ItemFactory(Generic[T], metaclass=SingletonMeta):
    """Dataset factory enables registering, creating, and listing datasets. It is a singleton class."""

    _items: dict[str, T]

    def __init__(self):
        self._items = {}

    def __call__(cls, *args, **kwargs):
        return super().__call__(*args, **kwargs)

    @classmethod
    def shared(cls, factory: str):
        """Get the shared instance of the factory

        Returns:
            ItemFactory: shared instance
        """
        return cls(singleton=factory)

    def register(self, name: str, item: T) -> None:
        """Register an item

        Args:
            name (str): Unique item name
            item (T): Item
        """
        self._items[name] = item

    def unregister(self, name: str) -> None:
        """Unregister an item

        Args:
            name (str): Item name
        """
        self._items.pop(name, None)

    def list(self) -> list[str]:
        """List registered items

        Returns:
            list[str]: item names
        """
        return list(self._items.keys())

    def get(self, name: str) -> T:
        """Get an item

        Args:
            name (str): Item name

        Returns:
            HKDataset: dataset
        """
        return self._items[name]

    def has(self, name: str) -> bool:
        """Check if an item is registered

        Args:
            name (str): Item name

        Returns:
            bool: True if dataset is registered
        """
        return name in self._items


def create_factory(factory: str, type: Type[T]) -> ItemFactory[T]:
    """Create a factory

    Args:
        factory (str): Factory name
        type (Type[T]): Item type

    Returns:
        ItemFactory[T]: factory
    """
    return ItemFactory[T].shared(factory)
