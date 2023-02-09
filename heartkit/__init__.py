from .utils import setup_logger

try:
    setup_logger(__name__)
    from importlib.metadata import version

    __version__ = version(__name__)
except ImportError:
    __version__ = "0.0.0"
