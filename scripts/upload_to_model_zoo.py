from pathlib import Path
import logging

import boto3
from argdantic import ArgParser, ArgField

logging.basicConfig(level=logging.INFO, format="%(name)s - %(levelname)s: %(message)s")
logger = logging.getLogger("hk.scripts")

parser = ArgParser()


@parser.command()
def upload_to_model_zoo(
    src: Path = ArgField("-s", description="Model path"),
    name: str = ArgField("-n", description="Model name"),
    task: str = ArgField("-t", description="Task", default="rhythm"),
    version: str = ArgField("-v", description="Version", default="latest"),
    adk: str = ArgField(description="ADK", default="heartkit"),
    bucket: str = ArgField("-b", description="Bucket", default="ambiqai-model-zoo"),
    assets: tuple[str, ...] = ArgField("-a", description="Assets", default=()),
    dryrun: bool = ArgField("-d", description="Dry run", default=False),
) -> int:
    """Upload model assets to model zoo on S3

    Args:
        src (Path): Model path
        name (str): Model name
        task (str, optional): Task. Defaults to 'rhythm'.
        version (str, optional): Version. Defaults to 'latest'.
        adk (str, optional): ADK. Defaults to 'heartkit'.
        bucket (str, optional): Bucket. Defaults to 'ambiqai-model-zoo'.
        assets (tuple[str,...], optional): Assets. Defaults to ().
        dryrun (bool, optional): Dry run. Defaults to False.

    Examples:
        ```bash
        python scripts/upload_to_model_zoo.py \
            --dryrun \
            -s ./results/model \
            -n model_name \
            -t rhythm \
            -v latest \
            -a configuration.json \
            -a model.keras \
            -a model.tflite \
            -a metrics.json \
            -a history.csv \
        ```
    """
    if not assets:
        assets = ("configuration.json", "model.keras", "model.tflite", "metrics.json", "history.csv")

    if not src.exists():
        logger.error(f"Model path {src} not found")
        return -1

    # Create an S3 client
    s3 = boto3.client("s3")

    dst_prefix = f"{adk}/{task}/{name}/{version}"
    # Upload all assets
    for asset in assets:
        file_path = src / asset
        if not file_path.exists():
            logger.error(f"Asset {file_path} not found")
            continue
        # END IF
        dst_key = f"{dst_prefix}/{asset}"
        logger.info(f"Uploading s3://{bucket}/{dst_key}")
        if not dryrun:
            s3.upload_file(str(file_path), bucket, dst_key)
    # END FOR
    return 0


if __name__ == "__main__":
    parser()

"""
python ./scripts/upload_to_model_zoo.py -t rhythm -s ./results/arr-2-eff-sm -n arr-2-eff-sm -v latest
python ./scripts/upload_to_model_zoo.py -t rhyhtm -s ./results/arr-4-eff-sm -n arr-4-eff-sm -v latest

python ./scripts/upload_to_model_zoo.py -t beat -s ./results/beat-2-eff-sm -n beat-2-eff-sm -v latest
python ./scripts/upload_to_model_zoo.py -t beat -s ./results/beat-3-eff-sm -n beat-3-eff-sm -v latest

python ./scripts/upload_to_model_zoo.py -t denoise -s ./results/den-ppg-tcn-sm -n den-ppg-tcn-sm -v latest
python ./scripts/upload_to_model_zoo.py -t denoise -s ./results/den-tcn-sm -n den-tcn-sm -v latest

python ./scripts/upload_to_model_zoo.py -t foundation -s ./results/fnd-eff-sm -n fnd-eff-sm -v latest

python ./scripts/upload_to_model_zoo.py -t segmentation -s ./results/seg-2-tcn-sm -n seg-2-tcn-sm -v latest
python ./scripts/upload_to_model_zoo.py -t segmentation -s ./results/seg-4-tcn-sm -n seg-4-tcn-sm -v latest
python ./scripts/upload_to_model_zoo.py -t segmentation -s ./results/seg-ppg-2-tcn-sm -n seg-ppg-2-tcn-sm -v latest
"""
