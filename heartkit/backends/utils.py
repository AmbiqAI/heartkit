import logging
import time
from typing import Optional

from serial.tools.list_ports import comports as list_ports
from serial.tools.list_ports_common import ListPortInfo

from .erpc.transport import SerialTransport

logger = logging.getLogger(__name__)


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
    vid_pid: Optional[str] = None, baudrate: Optional[int] = None, timeout: float = 10
) -> SerialTransport:
    """Create serial transport. Scans looking for port matching criteria.

    Args:
        vid_pid (Optional[str], optional): VID & PID. Defaults to None.
        baudrate (Optional[int], optional): Baudrate. Defaults to None.

    Raises:
        TimeoutError: Unable to find serial device within timeout

    Returns:
        SerialTransport: Serial device

    Examples:

    ```python
    import heartkit as hk
    transport = hk.backends.utils.get_serial_transport(vid_pid="51966:16385", baudrate=115200)

    transport.open()
    transport.write(b"Hello, World!")
    data = transport.read(13)
    print(data)
    transport.close()

    ```
    """
    port = None
    tic = time.time()
    while not port and (time.time() - tic) < timeout:
        port = _find_serial_device(vid_pid=vid_pid)
        if not port:
            time.sleep(0.5)
    if port is None:
        raise TimeoutError("Unable to locate serial port. Please verify connection")
    logger.debug(f"Found serial device @ {port.device}")
    return SerialTransport(port.device, baudrate=baudrate)
