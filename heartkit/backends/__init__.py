import neuralspot_edge as nse

from . import backend, utils, evb, pc

from .backend import HKInferenceBackend
from .evb import EvbBackend
from .pc import PcBackend

BackendFactory = nse.utils.create_factory("HKDemoBackend", HKInferenceBackend)

BackendFactory.register("pc", PcBackend)
BackendFactory.register("evb", EvbBackend)
