from multiprocessing import Process

from .defines import HeartDemoParams
from .evb import EvbHandler
from .pc import PcHandler
from .ui import ConsoleUi
from .utils import setup_logger

logger = setup_logger(__name__)


def demo(params: HeartDemoParams):
    """Run HeartKit demo

    Args:
        params (HeartDemoParams): Demo parameters
    """
    backend = None
    frontend = None
    try:
        if params.backend == "evb":
            backend = EvbHandler(params=params)
            backend.startup()
        elif params.backend == "pc":
            backend = PcHandler(params=params)
            backend.startup()
        else:
            raise ValueError("Invalid handler provided")

        if params.frontend == "console":
            console = ConsoleUi(params.rest_address)
            frontend = Process(target=console.run_forever, daemon=True)
            frontend.start()
        backend.run_forever()
    except KeyboardInterrupt:
        logger.info("Server stopping")
        if backend:
            backend.shutdown()
        if frontend:
            frontend.terminate()
    except Exception as err:  # pylint: disable=broad-except
        logger.exception(f"Unhandled error {err}")
