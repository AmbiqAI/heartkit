import os
from importlib.metadata import version

from . import cli, datasets, metrics, models, rpc, tasks, tflite
from .datasets import DatasetFactory, HKDataset
from .defines import (
    AugmentationParams,
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
from .tasks import HeartRate, HKBeat, HKRhythm, HKSegment, HKTask, TaskFactory
from .utils import setup_logger

__version__ = version(__name__)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
setup_logger(__name__)
