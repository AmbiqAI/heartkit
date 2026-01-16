"""
# heartKIT API

heartKIT is an AI Development Kit (ADK) that enables developers to easily train and deploy real-time __heart-monitoring__ models onto [Ambiq's family of ultra-low power SoCs](https://ambiq.com/soc/).
The kit provides a variety of datasets, efficient model architectures, and heart-related tasks.
In addition, heartKIT provides optimization and deployment routines to generate efficient inference models.
Finally, the kit includes a number of pre-trained models and task-level demos to showcase the capabilities.

"""

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
import helia_edge as helia

__version__ = version(__name__)

if "TF_CPP_MIN_LOG_LEVEL" not in os.environ:
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
helia.utils.setup_logger(__name__)
