import time
import logging
from typing import Optional, Tuple
from xml.dom import NotFoundErr
from serial.tools.list_ports_common import ListPortInfo
from serial.tools.list_ports import comports as list_ports
from serial.serialutil import SerialException
import numpy as np
import numpy.typing as npt
import erpc
from erpc.transport import SerialTransport
import pydantic_argparse
import plotext as plt
from rich.console import Console
from sklearn.utils import shuffle
from scipy.special import softmax
from .types import EcgDemoParams
from .utils import setup_logger
from . import datasets as ds
from .deploy import create_dataset
from .rpc import (
    GenericDataOperations_PcToEvb as gen_pc2evb,
    GenericDataOperations_EvbToPc as gen_evb2pc,
)

logger = logging.getLogger("ECGARR")
console = Console()


def _find_serial_device(
    vid_pid: Optional[str] = None,
    serial_number: Optional[str] = None,
    manufacturer: Optional[str] = None,
    product: Optional[str] = None,
) -> Optional[ListPortInfo]:
    """Find serial device based on optional fields.

    Args:
        vid_pid (Optional[str], optional): Vendor ID & product ID formatted as VID:PID. Defaults to None.
        serial_number (Optional[str], optional): Serial number. Defaults to None.
        manufacturer (Optional[str], optional): Manufacturer name. Defaults to None.
        product (Optional[str], optional): Product name. Defaults to None.

    Returns:
        ListPortInfo: Serial port info
    """
    ports = list_ports()
    for port in ports:
        if vid_pid and f"{port.vid}:{port.pid}" != vid_pid:
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
    vid_pid: Optional[str] = None, baudrate: Optional[int] = None
) -> SerialTransport:
    """Create serial transport to EVB. Scans looking for port for 30 seconds before giving up.

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
            raise NotFoundErr(
                "Unable to locate EVB serial port. Please verify connection"
            )
    logger.info(f"Found serial device @ {port.device}")
    return SerialTransport(port.device, baudrate=baudrate)


class DataServiceHandler(gen_evb2pc.interface.Ievb_to_pc):
    """Acts as delegate for eRPC generic data operations."""

    def __init__(self, params: EcgDemoParams) -> None:
        super().__init__()
        self.params = params
        self.test_x, self.test_y = self.load_test_data()
        self.setup_plot()
        # State
        self._sample_idx = 0
        self._frame_idx = 0
        self._plot_data = np.full(
            params.frame_size, fill_value=np.nan, dtype=np.float32
        )
        self._class_labels = ds.get_class_names(self.params.task)
        self._plot_results = np.zeros(len(self._class_labels))
        self._true_label = -1

    def load_test_data(self) -> Tuple[npt.ArrayLike, npt.ArrayLike]:
        """Load test data

        Returns:
            Tuple[npt.ArrayLike, npt.ArrayLike]: x,y
        """
        test_x, test_y = create_dataset(
            db_path=str(self.params.db_path),
            task=self.params.task,
            frame_size=self.params.frame_size,
            num_patients=200,
            samples_per_patient=10,
            normalize=False,
        )
        return shuffle(test_x, test_y)

    def setup_plot(self):
        """Setup plotting.
        """
        plt.theme("dark")

    def update_plot(self):
        """Update plot routine.
        """
        # plt.clt()
        # plt.cld()
        plt.clf()
        plt.subplots(1, 2)
        plt.subplot(1, 1).plotsize(3 * plt.tw() // 4, None)
        plt.title("Live Sensor Data")
        plt.plot(self._plot_data)
        plt.subplot(1, 2)
        title_label = ""
        if self._true_label != -1:
            title_label = f"[Y = {self._class_labels[self._true_label]}]"
        plt.title(f"Classification {title_label}")
        plt.bar(self._class_labels, self._plot_results)
        plt.show()

    def increment_frame_idx(self, inc: int):
        """Increment frame index and optional increment sample index

        Args:
            inc (int): increment
        """
        self._frame_idx += inc
        if self._frame_idx >= self.params.frame_size:
            logger.debug("Fetching next sample")
            self._frame_idx = 0
            self._sample_idx = (self._sample_idx + 1) % self.test_x.shape[0]

    def log_data(self, x):
        """ Log data to file. """
        with open(self.params.job_dir / "evb_data.csv", "a+", encoding="utf-8") as f:
            f.write("\n".join((f"{v:0.1f}" for v in x)) + "\n")

    def ns_rpc_data_sendBlockToPC(self, block: gen_pc2evb.common.dataBlock):
        if "SEND_SAMPLES" in block.description:
            x: npt.NDArray = np.frombuffer(block.buffer, dtype=np.float32)
            for v in x:
                self._plot_data[self._frame_idx] = v
                self.increment_frame_idx(1)
                if self._frame_idx % 25 == 0:
                    self.update_plot()
        if "SEND_RESULTS" in block.description:
            self._plot_results: npt.NDArray = softmax(
                np.frombuffer(block.buffer, dtype=np.float32)
            )
            self.update_plot()
        return 1

    def ns_rpc_data_fetchBlockFromPC(self, block):
        return 1

    def ns_rpc_data_computeOnPC(
        self, in_block: gen_evb2pc.common.dataBlock, result_block
    ):
        if "FETCH_SAMPLES" in in_block.description:
            num_samples = in_block.length
            fstart = self._frame_idx
            f_len = min(self.params.frame_size - self._frame_idx, num_samples)
            x = (
                self.test_x[self._sample_idx, fstart : fstart + f_len]
                .squeeze()
                .astype(np.float32)
            )
            self._true_label = self.test_y[self._sample_idx]
            self._plot_data[self._frame_idx : self._frame_idx + f_len] = x
            self.increment_frame_idx(f_len)
            if self._frame_idx % 30 == 0:
                self.update_plot()
            x = np.ascontiguousarray(x, dtype=np.float32).tobytes("C")

            result_block.value = gen_evb2pc.common.dataBlock(
                length=f_len,
                dType=gen_pc2evb.common.dataType.float32_e,
                description="ECG_SENSOR_RESPONSE",
                cmd=gen_evb2pc.common.command.generic_cmd,
                buffer=bytearray(x),
            )
        return 1

    def ns_rpc_data_remotePrintOnPC(self, msg):
        logger.info(f"{msg}")
        return 1


def evb_demo(params: EcgDemoParams):
    """EVB Demo

    Args:
        params (EcgDemoParams): Demo parameters
    """
    try:
        handler = DataServiceHandler(params=params)
        transport = get_serial_transport(
            vid_pid=params.vid_pid, baudrate=params.baudrate
        )
        service = gen_evb2pc.server.evb_to_pcService(handler)
        server = erpc.simple_server.SimpleServer(transport, erpc.basic_codec.BasicCodec)
        server.add_service(service)
        logger.info("Server running")
        server.run()
    except (KeyboardInterrupt, SerialException):
        logger.info("Server stopping")
    except Exception as err: # pylint: disable=broad-except
        logger.exception(f"Unhandled error {err}")


def create_parser():
    """Create CLI parser"""
    return pydantic_argparse.ArgumentParser(
        model=EcgDemoParams,
        prog="Heart arrhythmia EVB demo command",
        description="Demo heart arrhythmia model on EVB",
    )


if __name__ == "__main__":
    setup_logger("ECGARR")
    parser = create_parser()
    evb_demo(parser.parse_typed_args())
