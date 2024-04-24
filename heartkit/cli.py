import os
from typing import Type, TypeVar

from argdantic import ArgField, ArgParser
from pydantic import BaseModel

from .datasets import download_datasets
from .defines import (
    HKDemoParams,
    HKDownloadParams,
    HKExportParams,
    HKMode,
    HKTestParams,
    HKTrainParams,
)
from .tasks import TaskFactory
from .utils import setup_logger

logger = setup_logger(__name__)

cli = ArgParser()


B = TypeVar("B", bound=BaseModel)


def parse_content(cls: Type[B], content: str) -> B:
    """Parse file or raw content into Pydantic model.

    Args:
        cls (B): Pydantic model subclasss
        content (str): File path or raw content

    Returns:
        B: Pydantic model subclass instance
    """
    if os.path.isfile(content):
        with open(content, "r", encoding="utf-8") as f:
            content = f.read()

    return cls.model_validate_json(json_data=content)


@cli.command(name="run")
def _run(
    mode: HKMode = ArgField("-m", description="Mode"),
    task: str = ArgField("-t", description="Task"),
    config: str = ArgField("-c", description="File path or JSON content"),
):
    """HeartKit CLI"""

    logger.info(f"#STARTED MODE={mode} TASK={task}")

    if mode == HKMode.download:
        download_datasets(parse_content(HKDownloadParams, config))
        return

    if not TaskFactory.has(task):
        raise ValueError(f"Unknown task {task}")

    task_handler = TaskFactory.get(task)

    match mode:
        case HKMode.train:
            task_handler.train(parse_content(HKTrainParams, config))

        case HKMode.evaluate:
            task_handler.evaluate(parse_content(HKTestParams, config))

        case HKMode.export:
            task_handler.export(parse_content(HKExportParams, config))

        case HKMode.demo:
            task_handler.demo(parse_content(HKDemoParams, config))

        case _:
            logger.error("Error: Unknown command")
    # END MATCH

    logger.info(f"#FINISHED MODE={mode} TASK={task}")


def run():
    """Run CLI."""
    cli()


if __name__ == "__main__":
    run()
