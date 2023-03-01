import os

import pydantic_argparse
from pydantic import BaseModel, Field

from . import arrhythmia, beat, hrv, segmentation
from .datasets import download_datasets
from .defines import (
    HeartDemoParams,
    HeartDownloadParams,
    HeartExportParams,
    HeartKitMode,
    HeartTask,
    HeartTestParams,
    HeartTrainParams,
)
from .demo import evb_demo
from .utils import setup_logger

logger = setup_logger(__name__)


class CliArgs(BaseModel):
    """CLI arguments"""

    task: HeartTask = Field(default=HeartTask.segmentation)
    mode: HeartKitMode = Field(default=HeartKitMode.train)
    config: str = Field(description="JSON config file path or string")


def run(inputs: list[str] | None = None):
    """Main CLI app runner
    Args:
        inputs (list[str] | None, optional): App arguments. Defaults to CLI arguments.
    """
    parser = pydantic_argparse.ArgumentParser(
        model=CliArgs,
        prog="Heart Kit CLI",
        description="Heart Kit leverages AI for heart monitoring tasks.",
    )
    args = parser.parse_typed_args(inputs)

    logger.info(f"#STARTED {args.mode.value} model")

    match args.task:
        case HeartTask.rhythm:
            task_handler = arrhythmia
        case HeartTask.beat:
            task_handler = beat
        case HeartTask.segmentation:
            task_handler = segmentation
        case HeartTask.hr:
            task_handler = hrv
        case _:
            task_handler = None
    # END MATCH

    match args.mode:
        case HeartKitMode.download:
            download_datasets(
                params=(
                    HeartDownloadParams.parse_file(args.config)
                    if os.path.isfile(args.config)
                    else HeartDownloadParams.parse_raw(args.config)
                )
            )
        case HeartKitMode.train:
            assert task_handler
            task_handler.train_model(
                (
                    HeartTrainParams.parse_file(args.config)
                    if os.path.isfile(args.config)
                    else HeartTrainParams.parse_raw(args.config)
                )
            )

        case HeartKitMode.evaluate:
            assert task_handler
            task_handler.evaluate_model(
                params=(
                    HeartTestParams.parse_file(args.config)
                    if os.path.isfile(args.config)
                    else HeartTestParams.parse_raw(args.config)
                )
            )

        case HeartKitMode.export:
            assert task_handler
            task_handler.export_model(
                params=(
                    HeartExportParams.parse_file(args.config)
                    if os.path.isfile(args.config)
                    else HeartExportParams.parse_raw(args.config)
                )
            )

        case HeartKitMode.demo:
            evb_demo(
                task=args.task,
                params=(
                    HeartDemoParams.parse_file(args.config)
                    if os.path.isfile(args.config)
                    else HeartDemoParams.parse_raw(args.config)
                ),
            )

        case HeartKitMode.predict:
            raise NotImplementedError()
        case _:
            logger.error("Error: Unknown command")

    # END MATCH
    logger.info(f"#FINISHED {args.mode.value} model")


if __name__ == "__main__":
    run()
