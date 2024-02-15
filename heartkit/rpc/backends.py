import abc
import time
from enum import IntEnum

import erpc
import numpy as np
import numpy.typing as npt
import tensorflow as tf

from .. import tflite
from ..defines import HKDemoParams
from ..utils import setup_logger
from . import GenericDataOperations_PcToEvb as pc2evb
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

    def _send_binary(self, name: str, cmd: int, data: bytes, chunk_len: int = 128, delay: float = 0.01):
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
            description=name, dType=pc2evb.common.dataType.uint8_e, cmd=cmd, buffer=bytearray([0]), length=1
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
            model = fp.read()
        self._interpreter = tf.lite.Interpreter(model_content=model)
        self._interpreter.allocate_tensors()
        self._send_binary("MODEL", RpcCommands.SEND_MODEL, model)

    def set_inputs(self, inputs: npt.NDArray):
        inputs = inputs.copy()
        inputs = inputs.astype(np.float32)
        model_sig = self._interpreter.get_signature_runner()
        inputs_details = model_sig.get_input_details()
        input_name = list(inputs_details.keys())[0]
        input_details = inputs_details[input_name]
        logger.debug(input_details)

        input_scale: list[float] = input_details["quantization_parameters"]["scales"]
        input_zero_point: list[int] = input_details["quantization_parameters"]["zero_points"]
        inputs = inputs.reshape(input_details["shape_signature"])
        if len(input_scale) and len(input_zero_point):
            inputs = inputs / input_scale[0] + input_zero_point[0]
            inputs = inputs.astype(input_details["dtype"])

        self._send_binary("INPUT", RpcCommands.SEND_INPUT, np.ascontiguousarray(inputs).tobytes("C"))

    def perform_inference(self):
        self._send_binary("INFER", RpcCommands.PERFORM_INFERENCE, bytes([0]))
        while self._get_status() != 0:
            time.sleep(0.2)

    def get_outputs(self) -> npt.NDArray:
        data = self._fetch_binary("OUTPUT", RpcCommands.FETCH_OUTPUT)
        model_sig = self._interpreter.get_signature_runner()
        outputs_details = model_sig.get_output_details()
        output_name = list(outputs_details.keys())[0]
        output_details = outputs_details[output_name]
        logger.debug(output_details)
        output_scale: list[float] = output_details["quantization_parameters"]["scales"]
        output_zero_point: list[int] = output_details["quantization_parameters"]["zero_points"]
        outputs = np.frombuffer(data, dtype=output_details["dtype"])
        outputs = outputs.reshape(output_details["shape_signature"])
        if len(output_scale) and len(output_zero_point):
            outputs = outputs.astype(np.float32)
            outputs = (outputs - output_zero_point[0]) * output_scale[0]
        return outputs


class PcBackend(DemoBackend):
    """Demo backend for PC"""

    def __init__(self, params: HKDemoParams) -> None:
        super().__init__(params=params)
        self._inputs = None
        self._outputs = None
        self._model = None

    def _is_tf_model(self) -> bool:
        ext = self.params.model_file.split(".")[-1]
        return ext in ["h5", "hdf5", "keras", "tf"]

    def open(self):
        if self._is_tf_model():
            self._model = tflite.load_model(self.params.model_file)
        else:
            with open(self.params.model_file, "rb") as fp:
                self._model = fp.read()

    def close(self):
        self._model = None

    def set_inputs(self, inputs: npt.NDArray):
        self._inputs = inputs

    def perform_inference(self):
        if self._is_tf_model():
            self._outputs = self._model.predict(np.expand_dims(self._inputs, 0)).squeeze(0)
        else:
            self._outputs = tflite.predict_tflite(self._model, self._inputs)

    def get_outputs(self) -> npt.NDArray:
        return self._outputs
