import time
from enum import IntEnum

import helia_edge as helia
import numpy as np
import numpy.typing as npt

from ..defines import HKTaskParams
from . import GenericDataOperations_PcToEvb as pc2evb
from . import erpc
from .backend import HKInferenceBackend
from .utils import get_serial_transport


class RpcCommands(IntEnum):
    """RPC commands

    These commands are used to send data to the EVB.

    Attributes:
        SEND_MODEL (int): Send model to EVB
        SEND_INPUT (int): Send input data to EVB
        FETCH_OUTPUT (int): Fetch output data from EVB
        FETCH_STATUS (int): Fetch status from EVB
        PERFORM_INFERENCE (int): Perform inference on
    """

    SEND_MODEL = 0
    SEND_INPUT = 1
    FETCH_OUTPUT = 2
    FETCH_STATUS = 3
    PERFORM_INFERENCE = 4


class EvbBackend(HKInferenceBackend):
    def __init__(self, params: HKTaskParams) -> None:
        """EVB inference engine backend

        This backend leverages Ambiq SoCs to run inference on the edge.
        The model, inputs, and outputs are sent to the EVB using eRPC.

        By default, the backend will scan serial ports looking for the EVB.
        Therefore, the EVB must be connected and running prior to using this backend.

        Args:
            params (HKTaskParams): Task parameters
        """

        super().__init__(params=params)
        self._interpreter = None
        self._transport = None
        self._client = None

    def open(self):
        """Open connection to EVB

        The following steps are performed:
        1. Scan serial ports for EVB
        2. Connect to EVB
        3. Send model to EVB via eRPC

        """

        self._transport = get_serial_transport(vid_pid="51966:16401", baudrate=115200)
        client_manager = erpc.client.ClientManager(self._transport, erpc.basic_codec.BasicCodec)
        self._client = pc2evb.client.pc_to_evbClient(client_manager)
        self.send_model()

    def close(self):
        """Close connection to EVB

        This method will close the connection to the EVB.
        """
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
        """Send model to EVB

        This method sends the model to the EVB using eRPC. The TFLite flatbuffer will be read from disk
        and sent to the EVB.
        """

        with open(self.params.model_file, "rb") as fp:
            model_content = fp.read()
        self._interpreter = helia.converters.tflite.TfLiteKerasInterpreter(model_content=model_content)
        self._interpreter.compile()
        self._send_binary("MODEL", RpcCommands.SEND_MODEL, model_content)

    def set_inputs(self, inputs: npt.NDArray):
        """Set inputs for inference

        The inputs are flattened and sent to the EVB using eRPC.

        Args:
            inputs (npt.NDArray): Inputs for inference
        """
        inputs = self._interpreter.convert_input(inputs)
        self._send_binary("INPUT", RpcCommands.SEND_INPUT, np.ascontiguousarray(inputs).tobytes("C"))

    def perform_inference(self):
        """Perform inference

        This method sends the inference command to the EVB and waits for the inference to complete.
        """
        self._send_binary("INFER", RpcCommands.PERFORM_INFERENCE, bytes([0]))
        while self._get_status() != 0:
            time.sleep(0.2)

    def get_outputs(self) -> npt.NDArray:
        """Get outputs from inference

        The outputs are fetched from the EVB and converted to a numpy array.

        Returns:
            npt.NDArray: Outputs from inference
        """
        data = self._fetch_binary("OUTPUT", RpcCommands.FETCH_OUTPUT)
        outputs = np.frombuffer(data, dtype=self._interpreter._output_dtype)
        outputs = self._interpreter.convert_output(data)
        return outputs
