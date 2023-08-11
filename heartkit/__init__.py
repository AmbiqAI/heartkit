import os

from . import arrhythmia, beat, datasets, hrv, segmentation, tasks
from .utils import setup_logger

try:
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    setup_logger(__name__)
    from importlib.metadata import version

    __version__ = version(__name__)
except ImportError:
    __version__ = "0.0.0"
