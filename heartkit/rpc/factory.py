from .backends import DemoBackend

_backends: dict[str, DemoBackend] = {}


class BackendFactory:
    """Factory for registering, creating, and listing backends. It is a singleton class."""

    @staticmethod
    def register(name: str, backend: DemoBackend) -> None:
        """Register a backend

        Args:
            name (str): backend name
            backend (DemoBackend): backend
        """
        _backends[name] = backend

    @staticmethod
    def create(name: str, **kwargs) -> DemoBackend:
        """Create a backend

        Args:
            name (str): backend name

        Returns:
            DemoBackend: backend
        """
        return _backends[name](**kwargs)

    @staticmethod
    def list() -> list[str]:
        """List registered backends

        Returns:
            list[str]: backend names
        """
        return list(_backends.keys())

    @staticmethod
    def get(name: str) -> DemoBackend:
        """Get a backend

        Args:
            name (str): backend name

        Returns:
            DemoBackend: backend
        """
        return _backends[name]

    @staticmethod
    def has(name: str) -> bool:
        """Check if a backend is registered

        Args:
            name (str): backend name

        Returns:
            bool: True if backend is registered
        """
        return name in _backends
