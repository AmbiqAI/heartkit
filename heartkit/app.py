from pathlib import Path
from typing import List, Optional

import pydantic_argparse
from pydantic import BaseModel, Field

from . import datasets as ds
from . import demo, evaluate, export, train
from .types import (
    HeartDemoParams,
    HeartDownloadParams,
    HeartExportParams,
    HeartTestParams,
    HeartTrainParams,
)
from .utils import setup_logger

logger = setup_logger(__name__)


class AppCommandArgments(BaseModel):
    """App command arguments as configuration file."""

    config_file: Optional[Path] = Field(None, description="Configuration JSON file")


class AppArguments(BaseModel):
    """App CLI arguments"""

    download: Optional[AppCommandArgments] = Field(description="Fetch dataset")
    train: Optional[AppCommandArgments] = Field(description="Train model")
    evaluate: Optional[AppCommandArgments] = Field(description="Test model")
    export: Optional[AppCommandArgments] = Field(description="Export model")
    demo: Optional[AppCommandArgments] = Field(description="EVB demo")


def download_dataset(command: AppCommandArgments):
    """Download dataset CLI command.
    Args:
        command (AppCommandArgments): Command arguments
    """
    params = HeartDownloadParams.parse_file(command.config_file)
    logger.info("#STARTED downloading dataset")
    ds.download_datasets(params=params)
    logger.info("#FINISHED downloading dataset")


def train_model(command: AppCommandArgments):
    """Train model CLI command.
    Args:
        command (AppCommandArgments): Command arguments
    """
    params = HeartTrainParams.parse_file(command.config_file)
    logger.info("#STARTED training model")
    train.train_model(params=params)
    logger.info("#FINISHED training model")


def evaluate_model(command: AppCommandArgments):
    """Test model CLI command.
    Args:
        command (AppCommandArgments): Command arguments
    """

    params = HeartTestParams.parse_file(command.config_file)
    logger.info("#STARTED testing model")
    evaluate.evaluate_model(params=params)
    logger.info("#FINISHED testing model")


def export_model(command: AppCommandArgments):
    """Deploy model CLI command.
    Args:
        command (AppCommandArgments): Command arguments
    """
    params = HeartExportParams.parse_file(command.config_file)
    logger.info("#STARTED deploying model")
    export.export_model(params=params)
    logger.info("#FINISHED deploying model")


def evb_demo(command: AppCommandArgments):
    """EVB Demo CLI command.
    Args:
        command (AppCommandArgments): Command arguments
    """
    params = HeartDemoParams.parse_file(command.config_file)
    logger.info("#STARTED evb demo")
    demo.evb_demo(params=params)
    logger.info("#FINISHED evb demo")


def run(inputs: Optional[List[str]] = None):
    """Main CLI app runner
    Args:
        inputs (Optional[List[str]], optional): App arguments. Defaults to CLI arguments.
    """
    parser = pydantic_argparse.ArgumentParser(
        model=AppArguments,
        prog="Heart Kit Demo",
        description="Heart Kit demo",
    )
    args = parser.parse_typed_args(inputs)

    if args.download:
        download_dataset(args.download)
    elif args.train:
        train_model(args.train)
    elif args.evaluate:
        evaluate_model(args.evaluate)
    elif args.export:
        export_model(args.export)
    elif args.demo:
        evb_demo(args.demo)
    else:
        logger.error("Error: Unknown command")


if __name__ == "__main__":
    run()
