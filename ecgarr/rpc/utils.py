import time
import logging
from typing import Optional
from rich.console import Console
from erpc.transport import SerialTransport
from serial.tools.list_ports_common import ListPortInfo
from serial.tools.list_ports import comports as list_ports

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
            raise TimeoutError(
                "Unable to locate EVB serial port. Please verify connection"
            )
    logger.info(f"Found serial device @ {port.device}")
    return SerialTransport(port.device, baudrate=baudrate)
