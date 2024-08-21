import os
from importlib.metadata import version

from . import cli, datasets, models, backends, tasks
from .datasets import DatasetFactory, HKDataset, HKDataloader
from .defines import (
    QuantizationParams,
    HKTaskParams,
    HKMode,
    NamedParams,
)
from .models import ModelFactory
from .tasks import HKBeat, HKRhythm, HKSegment, HKTask, TaskFactory
from .backends import BackendFactory
import neuralspot_edge as nse

__version__ = version(__name__)

if "TF_CPP_MIN_LOG_LEVEL" not in os.environ:
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
nse.utils.setup_logger(__name__)
