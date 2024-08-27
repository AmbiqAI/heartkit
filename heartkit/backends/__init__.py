"""
# :material-engine: Backends API

This module provides the built-in backend inference engines.

Classes:
    HKInferenceBackend: Base class for all inference engines.
    EvbBackend: EVB inference engine.
    PcBackend: PC inference engine.

"""

import neuralspot_edge as nse

from . import backend, utils, evb, pc

from .backend import HKInferenceBackend
from .evb import EvbBackend
from .pc import PcBackend

BackendFactory = nse.utils.create_factory("HKDemoBackend", HKInferenceBackend)

BackendFactory.register("pc", PcBackend)
BackendFactory.register("evb", EvbBackend)
