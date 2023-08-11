import ctypes
import threading
import time
from enum import IntEnum, StrEnum
from typing import Generator

import erpc
import numpy as np
import numpy.typing as npt
from erpc.simple_server import ServerThread as RpcServerThread
from erpc.transport import SerialTransport
from serial.serialutil import SerialException

from neuralspot.rpc import GenericDataOperations_EvbToPc as gen_evb2pc
from neuralspot.rpc import GenericDataOperations_PcToEvb as gen_pc2evb
from neuralspot.rpc.utils import get_serial_transport

from ..datasets import IcentiaDataset, LudbDataset, SyntheticDataset
from .client import HKRestClient
from .defines import AppState, HeartDemoParams, HeartKitState, HKResult
from .utils import setup_logger

logger = setup_logger(__name__)


class RpcResponse(IntEnum):
    """RPC Response value"""

    SUCCESS = 0
    FAILURE = 1


class RpcBlockCommands(StrEnum):
    """RPC EVB block commands"""

    SEND_SAMPLES = "SEND_SAMPLES"
    SEND_MASK = "SEND_MASK"
    SEND_RESULTS = "SEND_RESULTS"
    FETCH_SAMPLES = "FETCH_SAMPLES"


class HKResultStruct(ctypes.Structure):
    """EVB struct for storing results."""

    # _pack_ = 1
    _fields_ = [
        ("heart_rate", ctypes.c_uint32),
        ("heart_rhythm", ctypes.c_uint32),
        ("num_norm_beats", ctypes.c_uint32),
        ("num_pac_beats", ctypes.c_uint32),
        ("num_pvc_beats", ctypes.c_uint32),
        ("num_noise_beats", ctypes.c_uint32),
        ("arrhythmia", ctypes.c_uint32),
    ]

    def to_pydantic(self) -> HKResult:
        """Convert to python HKResult"""
        return HKResult(
            heart_rate=self.heart_rate,
            heart_rhythm=self.heart_rhythm,
            num_norm_beats=self.num_norm_beats,
            num_pac_beats=self.num_pac_beats,
            num_pvc_beats=self.num_pvc_beats,
            num_noise_beats=self.num_noise_beats,
            arrhythmia=bool(self.arrhythmia),
        )


class EvbHandler(gen_evb2pc.interface.Ievb_to_pc):
    """EVB Handler. Acts as delegate for eRPC generic data operation to EVB."""

    def __init__(self, params: HeartDemoParams) -> None:
        super().__init__()

        self.params = params
        self.client = HKRestClient(addr=params.rest_address)

        # HeartKit state
        self.hk_state = HeartKitState(
            data_id=0,
            app_state=AppState.IDLE_STATE,
            data=params.frame_size * [0],
            seg_mask=params.frame_size * [0],
            results=HKResult(),
        )

        self.data_gen = self.create_data_generator()
        self._frame_idx = 0
        self._run = False

        # Internal RPC handler
        self._rpc = None
        self._transport: SerialTransport | None = None

    def create_data_generator(self) -> Generator[npt.NDArray[np.float32], None, None]:
        """Create data generator

        Returns:
            Generator[npt.NDArray[np.float32], None, None]: Data generator
        """
        data_handlers = dict(icentia11k=IcentiaDataset, synthetic=SyntheticDataset, ludb=LudbDataset)
        logger.info(f"Loading dataset {self.params.dataset}")
        DataHandler = data_handlers.get(self.params.dataset, LudbDataset)
        ds = DataHandler(
            ds_path=str(self.params.ds_path),
            frame_size=self.params.frame_size,
            target_rate=self.params.sampling_rate,
        )
        pt_gen = ds.uniform_patient_generator(ds.get_test_patient_ids())
        data_gen = ds.signal_generator(pt_gen, samples_per_patient=self.params.samples_per_patient)
        return data_gen

    def ns_rpc_data_sendBlockToPC(self, block: gen_pc2evb.common.dataBlock):
        """RPC callback handler"""
        if RpcBlockCommands.SEND_SAMPLES in block.description:
            x: list[float] = np.frombuffer(block.buffer, dtype=np.float32).tolist()
            xs = block.length  # Use block.length as block offset
            xe = xs + min(len(x), len(self.hk_state.data) - xs)
            self.hk_state.data[xs:xe] = x[: (xe - xs)]
            self._frame_idx = xs

        if RpcBlockCommands.SEND_RESULTS in block.description:
            self.hk_state.results = HKResultStruct.from_buffer_copy(block.buffer).to_pydantic()

        if RpcBlockCommands.SEND_MASK in block.description:
            x: list[int] = np.frombuffer(block.buffer, dtype=np.uint8).tolist()
            xs = block.length  # Use block.length as block offset
            xe = xs + min(len(x), len(self.hk_state.seg_mask) - xs)
            self.hk_state.seg_mask[xs:xe] = x[: (xe - xs)]
        return RpcResponse.SUCCESS

    def ns_rpc_data_fetchBlockFromPC(self, block):
        """RPC callback handler"""
        return RpcResponse.SUCCESS

    def ns_rpc_data_computeOnPC(self, in_block: gen_evb2pc.common.dataBlock, result_block):
        """RPC callback handler"""
        if RpcBlockCommands.FETCH_SAMPLES in in_block.description:
            xs = self._frame_idx
            xe = xs + min(in_block.length, len(self.hk_state.data) - xs)
            x = np.ascontiguousarray(self.hk_state.data[xs:xe], dtype=np.float32).tobytes("C")
            self._frame_idx = xe
            result_block.value = gen_evb2pc.common.dataBlock(
                length=xs,  # Use block.length as block offset
                dType=gen_pc2evb.common.dataType.float32_e,
                description="RESPONSE",
                cmd=gen_evb2pc.common.command.generic_cmd,
                buffer=bytearray(x),
            )
        return RpcResponse.SUCCESS

    def update_app_state(self, app_state: AppState):
        """Handle app update events."""
        if app_state == self.hk_state.app_state:
            return
        self.hk_state.app_state = app_state
        logger.debug(f"[EVB] State={self.hk_state.app_state}")
        if self.hk_state.app_state == AppState.IDLE_STATE:
            self.client.set_app_state(self.hk_state.app_state)
        elif self.hk_state.app_state == AppState.COLLECT_STATE:
            # Keep sending/receiving data samples
            self._frame_idx = 0
            self.hk_state.data = next(self.data_gen).squeeze().tolist()
            self.hk_state.data_id = (self.hk_state.data_id + 1) % (2**20)
            self.hk_state.seg_mask = len(self.hk_state.data) * [0]
            self.client.set_app_state(self.hk_state.app_state)
        elif self.hk_state.app_state == AppState.PREPROCESS_STATE:
            self.client.set_app_state(self.hk_state.app_state)
        elif self.hk_state.app_state == AppState.INFERENCE_STATE:
            self.client.set_app_state(self.hk_state.app_state)
        elif self.hk_state.app_state == AppState.DISPLAY_STATE:
            self.client.set_state(self.hk_state)
        elif self.hk_state.app_state == AppState.FAIL_STATE:
            pass  # Log error message
        # END

    def ns_rpc_data_remotePrintOnPC(self, msg: str):
        # If EVB app FSM updates
        state: AppState | None = next((s for s in AppState if s in msg), None)
        if state is not None:
            self.update_app_state(app_state=state)
        # Otherwise treat as log message
        else:
            logger.debug(f"[EVB] {msg}")
        return RpcResponse.SUCCESS

    def handle_thread_exception(self, args: threading.ExceptHookArgs):
        """Handle thread exceptions."""
        logger.exception(f"RPC Server error: {args.exc_value} {args.exc_type}")
        # Fatal if not serial exception- otherwise we assume EVB was reset/disconnected
        is_fatal = args.exc_type != SerialException
        if is_fatal:
            self._run = False
        self.stop_rpc()

    def start_rpc(self):
        """Start RPC"""
        self._rpc = None
        self._transport = None
        while not self._transport:
            try:
                if not self._run:
                    return
                self._transport = get_serial_transport(vid_pid=self.params.vid_pid, baudrate=self.params.baudrate)
            except TimeoutError:
                logger.warning("Unable to locate EVB device. Retrying in 5 secs...")
                self._transport = None
                time.sleep(5)
        self._rpc = RpcServerThread(self._transport, erpc.basic_codec.BasicCodec)
        self._rpc.add_service(gen_evb2pc.server.evb_to_pcService(self))
        threading.excepthook = self.handle_thread_exception
        self._rpc.start()

    def stop_rpc(self):
        """Stop RPC"""
        try:
            if self._rpc:
                self._rpc.stop()
        except Exception:  # pylint: disable=broad-exception-caught
            pass
        try:
            if self._transport:
                self._transport.close()
        except Exception:  # pylint: disable=broad-exception-caught
            pass
        self._rpc = None
        self._transport = None

    def startup(self):
        """Startup EVB backend"""
        self._run = True

    def run_forever(self):
        """Run backend"""
        while self._run:
            self.start_rpc()
            # Keep running unitl
            while self._rpc:
                time.sleep(0.5)
            # END WHILE
            time.sleep(0.5)
        # END WHILE

    def shutdown(self):
        """Shutdown EVB backend"""
        self._run = False
        self.stop_rpc()
