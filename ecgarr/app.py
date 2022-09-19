from pathlib import Path
from typing import Optional
import pydantic_argparse
from pydantic import BaseModel, Field
from .types import (
    EcgDownloadParams,
    EcgTrainParams,
    EcgTestParams,
    EcgDeployParams
)
from . import datasets as ds
from . import train
from . import evaluate
from . import demo
from . import deploy

class AppCommandArgments(BaseModel):
    config_file: Optional[Path] = Field(None, description='Configuration JSON file')

class AppArguments(BaseModel):
    download_dataset: Optional[AppCommandArgments] = Field(description="Fetch dataset")
    train_model: Optional[AppCommandArgments] = Field(description="Train model")
    test_model: Optional[AppCommandArgments] = Field(description="Test model")
    deploy_model: Optional[AppCommandArgments] = Field(description="Deploy model")
    evb_demo: Optional[AppCommandArgments] = Field(description="EVB demo")

def download_dataset(command: AppCommandArgments):
    params = EcgDownloadParams.parse_file(command.config_file)
    print("#STARTED downloading dataset")
    ds.download_datasets(params=params)
    print('#FINISHED downloading dataset')

def train_model(command: AppCommandArgments):
    params = EcgTrainParams.parse_file(command.config_file)
    print("#STARTED training model")
    train.train_model(params=params)
    print("#FINISHED training model")

def test_model(command: AppCommandArgments):
    params = EcgTestParams.parse_file(command.config_file)
    print("#STARTED testing model")
    evaluate.evaluate_model(params=params)
    print("#FINISHED testing model")

def deploy_model(command: AppCommandArgments):
    params = EcgDeployParams.parse_file(command.config_file)
    print("#STARTED deploying model")
    deploy.deploy_model(params=params)
    print("#FINISHED deploying model")

def evb_demo(command: AppCommandArgments):
    print("#STARTED evb demo")
    demo.evb_demo()
    print("#FINISHED evb demo")

if __name__ == "__main__":
    parser = pydantic_argparse.ArgumentParser(
        model=AppArguments,
        prog="ECG arrhythmia Demo",
        description="ECG arrhythmia demo"
    )
    args = parser.parse_typed_args()

    if args.download_dataset:
        download_dataset(args.download_dataset)
    elif args.train_model:
        train_model(args.train_model)
    elif args.test_model:
        test_model(args.test_model)
    elif args.deploy_model:
        deploy_model(args.deploy_model)
    elif args.evb_demo:
        evb_demo(args.evb_demo)
    else:
        print(f'Error: Unknown command')
