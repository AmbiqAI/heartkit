import neuralspot_edge as nse

from . import GenericDataOperations_EvbToPc as evb2pc
from . import GenericDataOperations_PcToEvb as pc2evb
from . import utils
from .backends import HKInferenceBackend, EvbBackend, PcBackend

BackendFactory = nse.utils.create_factory("HKDemoBackend", HKInferenceBackend)

BackendFactory.register("pc", PcBackend)
BackendFactory.register("evb", EvbBackend)
