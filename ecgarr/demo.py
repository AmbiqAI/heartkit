import logging
import pydantic_argparse
from .utils import setup_logger
from .types import EcgDemoParams

logger = logging.getLogger('ecgarr.deploy')

def evb_demo():
    setup_logger('ecgarr', str(args.job_dir))
    return
    # 1. Start RPC server
    # 2. Send N samples and receive N labels
    # 3. Display results

def create_parser():
    return pydantic_argparse.ArgumentParser(
        model=EcgDemoParams,
        prog="ECG Arrhythmia EVB Demo",
        description="ECG Arrhythmia Demo on EVB"
    )

if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_typed_args()
    evb_demo(args)
