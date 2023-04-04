from ..defines import HeartDemoParams
from ..utils import setup_logger
from .evb import EvbHandler
from .pc import PcHandler

logger = setup_logger(__name__)


def demo(params: HeartDemoParams):
    """Run HeartKit demo

    Args:
        params (HeartDemoParams): Demo parameters
    """
    handler = None
    try:
        if params.backend == "evb":
            handler = EvbHandler(params=params)
            handler.startup()
        elif params.backend == "pc":
            handler = PcHandler(params=params)
            handler.startup()
        else:
            raise ValueError("Invalid handler provided")
        handler.run_forever()
    except KeyboardInterrupt:
        logger.info("Server stopping")
        if handler:
            handler.shutdown()
    except Exception as err:  # pylint: disable=broad-except
        logger.exception(f"Unhandled error {err}")
