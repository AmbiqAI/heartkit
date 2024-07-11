import functools
import os
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Generator, Iterable, TypeVar

import boto3
import numpy as np
import numpy.typing as npt
import tensorflow as tf
from botocore import UNSIGNED
from botocore.client import Config
from tqdm import tqdm

from ..utils import compute_checksum, setup_logger

logger = setup_logger(__name__)


def create_dataset_from_data(x: npt.NDArray, y: npt.NDArray, spec: tuple[tf.TensorSpec]) -> tf.data.Dataset:
    """Helper function to create dataset from static data

    Args:
        x (npt.NDArray): Numpy data
        y (npt.NDArray): Numpy labels

    Returns:
        tf.data.Dataset: Dataset
    """
    return tf.data.Dataset.zip((tf.data.Dataset.from_tensor_slices(x), tf.data.Dataset.from_tensor_slices(y)))


T = TypeVar("T")
K = TypeVar("K")


def buffered_generator(generator: Generator[T, None, None], buffer_size: int) -> Generator[list[T], None, None]:
    """Buffer the elements yielded by a generator. New elements replace the oldest elements in the buffer.

    Args:
        generator (Generator[T]): Generator object.
        buffer_size (int): Number of elements in the buffer.

    Returns:
        Generator[list[T], None, None]: Yields a buffer.
    """
    buffer = []
    for e in generator:
        buffer.append(e)
        if len(buffer) == buffer_size:
            break
    yield buffer
    for e in generator:
        buffer = buffer[1:] + [e]
        yield buffer


def uniform_id_generator(
    ids: Iterable[T],
    repeat: bool = True,
    shuffle: bool = True,
) -> Generator[T, None, None]:
    """Simple generator that yields ids in a uniform manner.

    Args:
        ids (pt.ArrayLike): Array of ids
        repeat (bool, optional): Whether to repeat generator. Defaults to True.
        shuffle (bool, optional): Whether to shuffle ids.. Defaults to True.

    Returns:
        Generator[T, None, None]: Generator
    Yields:
        T: Id
    """
    ids = np.copy(ids)
    while True:
        if shuffle:
            np.random.shuffle(ids)
        yield from ids
        if not repeat:
            break
        # END IF
    # END WHILE


def random_id_generator(
    ids: Iterable[T],
    weights: list[int] | None = None,
) -> Generator[T, None, None]:
    """Simple generator that yields ids in a random manner.

    Args:
        ids (pt.ArrayLike): Array of ids
        weights (list[int], optional): Weights for each id. Defaults to None.

    Returns:
        Generator[T, None, None]: Generator

    Yields:
        T: Id
    """
    while True:
        yield random.choice(ids)
    # END WHILE


def transform_dataset_pipeline(
    ds: tf.data.Dataset,
    buffer_size: int | None = None,
    batch_size: int | None = None,
    prefetch_size: int | None = None,
) -> tf.data.Dataset:
    """Transform dataset pipeline

    Args:
        ds (tf.data.Dataset): Dataset
        buffer_size (int | None, optional): Buffer size. Defaults to None.
        batch_size (int | None, optional): Batch size. Defaults to None.
        prefetch_size (int | None, optional): Prefetch size. Defaults to None.

    Returns:
        tf.data.Dataset: Transformed dataset
    """
    if buffer_size is not None:
        ds = ds.shuffle(
            buffer_size=buffer_size,
            reshuffle_each_iteration=True,
        )
    if batch_size is not None:
        ds = ds.batch(
            batch_size=batch_size,
            drop_remainder=False,
        )
    if prefetch_size is not None:
        ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    return ds


def create_interleaved_dataset_from_generator(
    data_generator: Callable[[Generator[T, None, None]], Generator[K, None, None]],
    id_generator: Callable[[list[T]], Generator[T, None, None]],
    ids: list[T],
    spec: tuple[tf.TensorSpec, tf.TensorSpec],
    preprocess: Callable[[K], K] | None = None,
    num_workers: int = 4,
) -> tf.data.Dataset:
    """Create TF dataset pipeline by interleaving multiple workers across ids

    The id_generator is used to generate ids for each worker.
    The data_generator is used to generate data for each id.

    Args:
        data_generator (Callable[[Generator[T, None, None]], Generator[K, None, None]]): Data generator
        id_generator (Callable[[list[T]], Generator[T, None, None]]): Id generator
        ids (list[T]): List of ids
        spec (tuple[tf.TensorSpec, tf.TensorSpec]): Tensor spec
        preprocess (Callable[[K], K] | None, optional): Preprocess function. Defaults to None.
        num_workers (int, optional): Number of workers. Defaults to 4.

    Returns:
        tf.data.Dataset: Dataset
    """

    def split_generator(split_ids: list[T]) -> tf.data.Dataset:
        """Split generator per worker"""

        def ds_gen():
            """Worker generator routine"""
            split_id_generator = id_generator(split_ids)
            return map(preprocess, data_generator(split_id_generator))

        return tf.data.Dataset.from_generator(
            ds_gen,
            output_signature=spec,
        )

    # END IF

    num_workers = min(num_workers, len(ids))
    split = len(ids) // num_workers
    logger.info(f"Splitting {len(ids)} ids into {num_workers} workers with {split} ids each")
    ds_splits = [split_generator(ids[i * split : (i + 1) * split]) for i in range(num_workers)]

    # Create TF datasets (interleave workers)
    ds = tf.data.Dataset.from_tensor_slices(ds_splits)

    ds = ds.interleave(
        lambda x: x,
        cycle_length=num_workers,
        deterministic=False,
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    return ds


def download_s3_object(
    item: dict[str, str],
    dst: Path,
    bucket: str,
    client: boto3.client,
    checksum: str = "size",
):
    """Download an object from S3

    Args:
        object (dict[str, str]): Object metadata
        dst (Path): Destination path
        bucket (str): Bucket name
        client (boto3.client): S3 client
        checksum (str, optional): Checksum type. Defaults to "size".
    """

    # Is a directory, skip
    if item["Key"].endswith("/"):
        print(f"Creating dir {dst}")
        os.makedirs(dst, exist_ok=True)
        return

    if not dst.is_file():
        pass
    elif checksum == "size":
        if dst.stat().st_size == item["Size"]:
            print(".", end="")
            return
    elif checksum == "md5":
        etag = item["ETag"]
        checksum_type = item.get("ChecksumAlgorithm", ["md5"])[0]
        calculated_checksum = compute_checksum(dst, checksum)
        if etag == calculated_checksum and checksum_type.lower() == "md5":
            return
    # END IF

    client.download_file(
        Bucket=bucket,
        Key=item["Key"],
        Filename=str(dst),
    )


def download_s3_objects(
    bucket: str,
    prefix: str,
    dst: Path,
    checksum: str = "size",
    progress: bool = True,
    num_workers: int | None = None,
    config: Config | None = Config(signature_version=UNSIGNED),
):
    """Download all objects in a S3 bucket with a given prefix

    Args:
        bucket (str): Bucket name
        prefix (str): Prefix to filter objects
        dst (Path): Destination directory
        checksum (str, optional): Checksum type. Defaults to "size".
        progress (bool, optional): Show progress bar. Defaults to True.
        num_workers (int | None, optional): Number of workers. Defaults to None.
        config (Config | None, optional): Boto3 config. Defaults to Config(signature_version=UNSIGNED).

    """

    session = boto3.Session()
    client = session.client("s3", config=config)

    # Fetch all objects in the bucket with the given prefix
    items = []
    fetching = True
    next_token = None
    while fetching:
        if next_token is None:
            response = client.list_objects_v2(Bucket=bucket, Prefix=prefix)
        else:
            response = client.list_objects_v2(Bucket=bucket, Prefix=prefix, ContinuationToken=next_token)
        items.extend(response["Contents"])
        next_token = response.get("NextContinuationToken", None)
        fetching = next_token is not None
    # END WHILE

    logger.info(f"Found {len(items)} objects in {bucket}/{prefix}")

    os.makedirs(dst, exist_ok=True)

    func = functools.partial(download_s3_object, bucket=bucket, client=client, checksum=checksum)

    pbar = tqdm(total=len(items), unit="objects") if progress else None

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = (
            executor.submit(
                func,
                item,
                dst / item["Key"],
            )
            for item in items
        )
        for future in as_completed(futures):
            err = future.exception()
            if err:
                logger.exception("Failed on file")
            if pbar:
                pbar.update(1)
        # END FOR
    # END WITH
