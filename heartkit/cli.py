"""
# :octicons-terminal-24: heartKIT CLI API

The heartKIT CLI provides a command-line interface to interact with the heartKIT library.

```bash
$ heartkit --help

heartKIT CLI Options:
    --task [segmentation, rhythm, beat, denoise]
    --mode [download, feature, train, evaluate, export, demo]
    --config ["./path/to/config.json", or '{"raw: "json"}']
```

"""

import os
from typing import Type, TypeVar

from argdantic import ArgField, ArgParser
from pydantic import BaseModel
import helia_edge as helia

from .defines import HKMode, HKTaskParams
from .tasks import TaskFactory

B = TypeVar("B", bound=BaseModel)

logger = helia.utils.setup_logger(__name__)

parser = ArgParser()


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


@parser.command()
def run(
    mode: HKMode = ArgField("-m", description="Mode", default=HKMode.train),
    task: str = ArgField("-t", description="Task", default="rhythm"),
    config: str = ArgField("-c", description="File path or JSON content", default="{}"),
):
    """heartKIT CLI entry point.

    Args:
        mode (HKMode, optional): Mode. Defaults to HKMode.train.
        task (str, optional): Task. Defaults to "rhythm".
        config (str, optional): File path or JSON content. Defaults to "{}".
    """

    logger.info(f"#STARTED MODE={mode} TASK={task}")

    if not TaskFactory.has(task):
        raise ValueError(f"Unknown task {task}")

    task_handler = TaskFactory.get(task)

    params = parse_content(HKTaskParams, config)

    match mode:
        case HKMode.download:
            task_handler.download(params)

        case HKMode.train:
            task_handler.train(params)

        case HKMode.evaluate:
            task_handler.evaluate(params)

        case HKMode.export:
            task_handler.export(params)

        case HKMode.demo:
            task_handler.demo(params)

        case _:
            logger.error("Error: Unknown command")
    # END MATCH

    logger.info(f"#FINISHED MODE={mode} TASK={task}")


def main():
    parser()


if __name__ == "__main__":
    main()
