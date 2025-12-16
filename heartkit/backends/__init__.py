"""
# :material-engine: Backends API

This module provides the built-in backend inference engines.

Classes:
    HKInferenceBackend: Base class for all inference engines.
    EvbBackend: EVB inference engine.
    PcBackend: PC inference engine.

"""

import helia_edge as helia

from . import backend, utils, evb, pc

from .backend import HKInferenceBackend
from .evb import EvbBackend
from .pc import PcBackend

BackendFactory = helia.utils.create_factory("HKDemoBackend", HKInferenceBackend)

BackendFactory.register("pc", PcBackend)
BackendFactory.register("evb", EvbBackend)
