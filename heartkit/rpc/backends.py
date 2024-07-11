import abc
import time
from enum import IntEnum

import keras_edge as kedge
import numpy as np
import numpy.typing as npt

from ..defines import HKDemoParams
from ..utils import setup_logger
from . import GenericDataOperations_PcToEvb as pc2evb
from . import erpc
from .utils import get_serial_transport

logger = setup_logger(__name__)


class RpcCommands(IntEnum):
    """RPC commands"""

    SEND_MODEL = 0
    SEND_INPUT = 1
    FETCH_OUTPUT = 2
    FETCH_STATUS = 3
    PERFORM_INFERENCE = 4


class DemoBackend(abc.ABC):
    """Demo backend base class"""

    def __init__(self, params: HKDemoParams) -> None:
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


class EvbBackend(DemoBackend):
    """Demo backend for EVB"""

    def __init__(self, params: HKDemoParams) -> None:
        super().__init__(params=params)
        self._interpreter = None
        self._transport = None
        self._client = None

    def open(self):
        self._transport = get_serial_transport(vid_pid="51966:16385", baudrate=115200)
        client_manager = erpc.client.ClientManager(self._transport, erpc.basic_codec.BasicCodec)
        self._client = pc2evb.client.pc_to_evbClient(client_manager)
        self.send_model()

    def close(self):
        self._transport.close()
        self._transport = None
        self._client = None

    def _send_binary(
        self,
        name: str,
        cmd: int,
        data: bytes,
        chunk_len: int = 128,
        delay: float = 0.01,
    ):
        """Send binary data to EVB"""
        for i in range(0, len(data), chunk_len):
            buffer = data[i : i + chunk_len]
            self._client.ns_rpc_data_sendBlockToEVB(
                pc2evb.common.dataBlock(
                    description=name,
                    dType=pc2evb.common.dataType.uint8_e,
                    cmd=cmd,
                    buffer=buffer,
                    length=len(data),  # Send full length
                )
            )
            time.sleep(delay)
        # END FOR

    def _fetch_binary(self, name: str, cmd: int, delay: float = 0.01) -> bytearray:
        """Fetch binary data from EVB"""
        ref_block = erpc.Reference()
        block = pc2evb.common.dataBlock(
            description=name,
            dType=pc2evb.common.dataType.uint8_e,
            cmd=cmd,
            buffer=bytearray([0]),
            length=1,
        )
        self._client.ns_rpc_data_computeOnEVB(block, ref_block)
        data = bytearray(ref_block.value.length)
        offset = len(ref_block.value.buffer)
        data[:offset] = ref_block.value.buffer[:]

        # Fetch remaining
        while offset < ref_block.value.length:
            self._client.ns_rpc_data_computeOnEVB(block, ref_block)
            data[offset : offset + len(ref_block.value.buffer)] = ref_block.value.buffer[:]
            offset += len(ref_block.value.buffer)
        # END WHILE
        time.sleep(delay)
        return data

    def _get_status(self) -> int:
        status = self._fetch_binary("STATE", RpcCommands.FETCH_STATUS)
        return status[0]

    def send_model(self):
        """Send model to EVB"""

        with open(self.params.model_file, "rb") as fp:
            model_content = fp.read()
        self._interpreter = kedge.converters.tflite.TfLiteKerasInterpreter(model_content=model_content)
        self._interpreter.compile()
        self._send_binary("MODEL", RpcCommands.SEND_MODEL, model_content)

    def set_inputs(self, inputs: npt.NDArray):
        inputs = self._interpreter.convert_input(inputs)
        self._send_binary("INPUT", RpcCommands.SEND_INPUT, np.ascontiguousarray(inputs).tobytes("C"))

    def perform_inference(self):
        self._send_binary("INFER", RpcCommands.PERFORM_INFERENCE, bytes([0]))
        while self._get_status() != 0:
            time.sleep(0.2)

    def get_outputs(self) -> npt.NDArray:
        data = self._fetch_binary("OUTPUT", RpcCommands.FETCH_OUTPUT)
        outputs = np.frombuffer(data, dtype=self._interpreter._output_dtype)
        outputs = self._interpreter.convert_output(data)
        return outputs


class PcBackend(DemoBackend):
    """Demo backend for PC"""

    def __init__(self, params: HKDemoParams) -> None:
        super().__init__(params=params)
        self._inputs = None
        self._outputs = None
        self._model = None

    def _is_tf_model(self) -> bool:
        ext = self.params.model_file.suffix
        return ext in [".h5", ".hdf5", ".keras", ".tf"]

    def open(self):
        if self._is_tf_model():
            self._model = kedge.models.load_model(self.params.model_file)
        else:
            with open(self.params.model_file, "rb") as fp:
                model_content = fp.read()
            self._model = kedge.interpreters.tflite.TfLiteKerasInterpreter(model_content=model_content)

    def close(self):
        self._model = None

    def set_inputs(self, inputs: npt.NDArray):
        self._inputs = inputs

    def perform_inference(self):
        if self._is_tf_model():
            self._outputs = self._model.predict(np.expand_dims(self._inputs, 0), verbose=0).squeeze(0)
        else:
            self._outputs = self._model.predict(self._inputs)

    def get_outputs(self) -> npt.NDArray:
        return self._outputs
