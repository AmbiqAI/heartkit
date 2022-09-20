import logging
import pydantic_argparse
from .types import EcgDemoParams
from .utils import setup_logger
logger = logging.getLogger('ecgarr.deploy')

def evb_demo():
    return
    # 1. Start RPC server
    # 2. Wait for data request -> send data
    # 3. Wait for classification result -> store result
    # 4. Display data and result -> [2]


def create_parser():
    return pydantic_argparse.ArgumentParser(
        model=EcgDemoParams,
        prog="ECG Arrhythmia EVB Demo Command",
        description="Demo ECG arrhythmia model on EVB"
    )

if __name__ == '__main__':
    """ Run ecgarr.demo as CLI. """
    setup_logger('ecgarr')
    parser = create_parser()
    args = parser.parse_typed_args()
    evb_demo(args)
