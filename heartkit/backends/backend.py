import abc

import helia_edge as helia
import numpy.typing as npt

from ..defines import HKTaskParams

logger = helia.utils.setup_logger(__name__)


class HKInferenceBackend(abc.ABC):
    def __init__(self, params: HKTaskParams) -> None:
        """Backend inference engine base class

        Args:
            params (HKTaskParams): Task parameters
        """
        self.params = params

    def open(self):
        """Open backend"""
        raise NotImplementedError

    def close(self):
        """Close backend"""
        raise NotImplementedError

    def set_inputs(self, inputs: npt.NDArray):
        """Set inputs"""
        raise NotImplementedError

    def perform_inference(self):
        """Perform inference"""
        raise NotImplementedError

    def get_outputs(self) -> npt.NDArray:
        """Get outputs"""
        raise NotImplementedError
