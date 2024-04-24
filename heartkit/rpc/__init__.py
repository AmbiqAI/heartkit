from . import GenericDataOperations_EvbToPc as evb2pc
from . import GenericDataOperations_PcToEvb as pc2evb
from . import utils
from .backends import DemoBackend, EvbBackend, PcBackend
from .factory import BackendFactory

BackendFactory.register("pc", PcBackend)
BackendFactory.register("evb", EvbBackend)
