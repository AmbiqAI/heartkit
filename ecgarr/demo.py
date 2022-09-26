import time
import logging
from typing import Optional
from xml.dom import NotFoundErr
from serial.tools.list_ports_common import ListPortInfo
from serial.tools.list_ports import comports as list_ports
import numpy as np
import erpc
from erpc.transport import SerialTransport
import pydantic_argparse
from rich.console import Console
from sklearn.utils import shuffle
from .types import EcgDemoParams
from .utils import setup_logger
from .deploy import create_dataset
from .rpc import (
    GenericDataOperations_PcToEvb as gen_pc2evb,
    GenericDataOperations_EvbToPc as gen_evb2pc
)

logger = logging.getLogger('ECGARR')
console = Console()

def _find_serial_device(
        vid_pid: Optional[str] = None,
        serial_number: Optional[str] = None,
        manufacturer: Optional[str] = None,
        product: Optional[str] = None
        ) -> Optional[ListPortInfo]:
    """ Find serial device based on optional fields.

    Args:
        vid_pid (Optional[str], optional): Vendor ID & product ID formatted as VID:PID. Defaults to None.
        serial_number (Optional[str], optional): Serial number. Defaults to None.
        manufacturer (Optional[str], optional): Manufacturer name. Defaults to None.
        product (Optional[str], optional): Product name. Defaults to None.

    Returns:
        _type_: _description_
    """
    ports = list_ports()
    for port in ports:
        if vid_pid and f'{port.vid}:{port.pid}' != vid_pid:
            continue
        if serial_number and port.serial_number != serial_number:
            continue
        if manufacturer and port.manufacturer.upper() != manufacturer.upper():
            continue
        if product and port.product.upper() != product.upper():
            continue
        return port
    return None

def get_serial_transport(
    vid_pid: Optional[str] = None,
    baudrate: Optional[int] = None
) -> SerialTransport:
    """ Create serial transport to EVB. Scans looking for port for 30 seconds before giving up.

    Args:
        vid_pid (Optional[str], optional): VID & PID. Defaults to None.
        baudrate (Optional[int], optional): Baudrate. Defaults to None.

    Raises:
        NotFoundErr: Unable to find serial device

    Returns:
        SerialTransport: Serial device
    """
    port = None
    with console.status("[bold green] Searching for EVB  port..."):
        tic = time.time()
        while not port and (time.time() - tic) < 30:
            port = _find_serial_device(vid_pid=vid_pid)
            if not port:
                time.sleep(0.5)
    if port is None:
        raise NotFoundErr('Unable to locate EVB serial port. Please verify connection')
    logger.info(f'Found serial device @ {port.device}')
    return SerialTransport(port.device, baudrate=baudrate)

class DataServiceHandler(gen_evb2pc.interface.Ievb_to_pc):
    """ Acts as delegate for eRPC generic data operations. """
    def __init__(self, params: EcgDemoParams) -> None:
        super().__init__()
        self.params = params
        self.test_x, self.test_y = create_dataset(
            db_path=str(params.db_path),
            task=params.task,
            frame_size=params.frame_size,
            num_patients=200,
            samples_per_patient=10
        )
        self.test_x, self.test_y = shuffle(self.test_x,  self.test_y)
        # State
        self._sample_idx = 0
        self._frame_idx = 0

    def ns_rpc_data_sendBlockToPC(self, block: gen_pc2evb.common.dataBlock):
        return 1

    def ns_rpc_data_fetchBlockFromPC(self, block):
        print("Got a ns_rpc_data_fetchBlockFromPC call.")
        return 1

    def ns_rpc_data_computeOnPC(self, in_block: gen_evb2pc.common.dataBlock, result_block):
        if 'ECG' in in_block.description:
            num_samples = in_block.length
            fstart = self._frame_idx
            f_len = min(self.params.frame_size - self._frame_idx, num_samples)
            x = self.test_x[self._sample_idx, fstart:fstart+f_len].squeeze().astype(np.float32)
            x = np.ascontiguousarray(x, dtype=np.float32)
            x = x.tobytes('C')
            self._frame_idx += f_len
            if self._frame_idx >= self.params.frame_size:
                logger.info(f'Label was {self.test_y[self._sample_idx]}')
                logger.info('Grabbing next sample')
                self._frame_idx = 0
                self._sample_idx = (self._sample_idx+1)%self.test_x.shape[0]
            result_block.value = gen_evb2pc.common.dataBlock(
                length = f_len,
                dType = gen_pc2evb.common.dataType.float32_e,
                description = "ECG_SENSOR_RESPONSE",
                cmd = gen_evb2pc.common.command.generic_cmd,
                buffer = bytearray(x),
            )
        return 1

    def ns_rpc_data_remotePrintOnPC(self, msg):
        logger.info(f"{msg}")
        return 1

def evb_demo(params: EcgDemoParams):
    """ EVB Demo

    Args:
        params (EcgDemoParams): Demo parameters
    """
    try:
        handler = DataServiceHandler(params=params)
        transport = get_serial_transport(vid_pid=params.vid_pid, baudrate=params.baudrate)
        service = gen_evb2pc.server.evb_to_pcService(handler)
        server = erpc.simple_server.SimpleServer(transport, erpc.basic_codec.BasicCodec)
        server.add_service(service)
        logger.info("Server running")
        server.run()
    except KeyboardInterrupt:
        pass # Allow user to stop demo nicely



def create_parser():
    """ Create CLI parser """
    return pydantic_argparse.ArgumentParser(
        model=EcgDemoParams,
        prog="Heart arrhythmia EVB demo command",
        description="Demo heart arrhythmia model on EVB"
    )

if __name__ == '__main__':
    setup_logger('ECGARR')
    parser = create_parser()
    evb_demo(parser.parse_typed_args())
