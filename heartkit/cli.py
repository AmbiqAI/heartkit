import os
from typing import Type, TypeVar

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
from .demo.demo import demo
from .utils import setup_logger

logger = setup_logger(__name__)


class CliArgs(BaseModel):
    """CLI arguments"""

    task: HeartTask = Field(default=HeartTask.segmentation)
    mode: HeartKitMode = Field(default=HeartKitMode.train)
    config: str = Field(description="JSON config file path or string")


B = TypeVar("B", bound=BaseModel)


def parse_content(cls: Type[B], content: str) -> B:
    """Parse file or raw content into Pydantic model.

    Args:
        cls (B): Pydantic model subclasss
        content (str): File path or raw content

    Returns:
        B: Pydantic model subclass instance
    """
    return (
        cls.parse_file(content) if os.path.isfile(content) else cls.parse_raw(content)
    )


def run(inputs: list[str] | None = None):
    """Main CLI app runner
    Args:
        inputs (list[str] | None, optional): App arguments. Defaults to CLI arguments.
    """
    parser = pydantic_argparse.ArgumentParser(
        model=CliArgs,
        prog="HeartKit CLI",
        description="HeartKit leverages AI for heart monitoring tasks.",
    )
    args = parser.parse_typed_args(inputs)

    logger.info(f"#STARTED {args.mode.value} model")

    if args.mode == HeartKitMode.download:
        download_datasets(parse_content(HeartDownloadParams, args.config))
        return

    match args.task:
        case HeartTask.arrhythmia:
            task_handler = arrhythmia
        case HeartTask.beat:
            task_handler = beat
        case HeartTask.segmentation:
            task_handler = segmentation
        case HeartTask.hrv:
            task_handler = hrv
        case _:
            raise NotImplementedError()
    # END MATCH

    match args.mode:
        case HeartKitMode.train:
            task_handler.train_model(parse_content(HeartTrainParams, args.config))

        case HeartKitMode.evaluate:
            task_handler.evaluate_model(parse_content(HeartTestParams, args.config))

        case HeartKitMode.export:
            task_handler.export_model(parse_content(HeartExportParams, args.config))

        case HeartKitMode.demo:
            demo(params=parse_content(HeartDemoParams, args.config))

        case HeartKitMode.predict:
            raise NotImplementedError()

        case _:
            logger.error("Error: Unknown command")

    # END MATCH

    logger.info(f"#FINISHED {args.mode.value} model")


if __name__ == "__main__":
    run()
