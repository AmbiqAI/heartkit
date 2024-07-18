import os
from importlib.metadata import version

from . import cli, datasets, metrics, models, rpc, tasks
from .datasets import DatasetFactory, HKDataset
from .defines import (
    AugmentationParams,
    QuantizationParams,
    DatasetParams,
    HKDemoParams,
    HKDownloadParams,
    HKExportParams,
    HKMode,
    HKTestParams,
    HKTrainParams,
    PreprocessParams,
)
from .models import ModelFactory
from .tasks import HKBeat, HKRhythm, HKSegment, HKTask, TaskFactory
from .utils import setup_logger, silence_tensorflow

__version__ = version(__name__)

if "TF_CPP_MIN_LOG_LEVEL" not in os.environ:
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
setup_logger(__name__)
