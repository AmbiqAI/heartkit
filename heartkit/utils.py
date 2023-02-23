import gzip
import logging
import os
import pickle
import random

import numpy as np
import requests
import tensorflow as tf
from rich.logging import RichHandler
from tqdm import tqdm


def set_random_seed(seed: int | None = None) -> int:
    """Set random seed across libraries: TF, Numpy, Python

    Args:
        seed (int | None, optional): Random seed state to use. Defaults to None.

    Returns:
        int: Random seed
    """
    seed = seed or np.random.randint(2**16)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    return seed


def buffered_generator(generator, buffer_size: int):
    """Buffer the elements yielded by a generator. New elements replace the oldest elements in the buffer.

    Args:
        generator (Generator): Generator object.
        buffer_size (int): Number of elements in the buffer.

    Returns:
        Generator: Yields a buffer.
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


def load_pkl(file: str, compress: bool = True):
    """Load pickled file.

    Args:
        file (str): File path (.pkl)
        compress (bool, optional): If file is compressed. Defaults to True.

    Returns:
        Any: Contents of pickle
    """
    if compress:
        with gzip.open(file, "rb") as fh:
            return pickle.load(fh)
    else:
        with open(file, "rb") as fh:
            return pickle.load(fh)


def save_pkl(file: str, compress: bool = True, **kwargs):
    """Save python objects into pickle file.

    Args:
        file (str): File path (.pkl)
        compress (bool, optional): Whether to compress file. Defaults to True.
    """
    if compress:
        with gzip.open(file, "wb") as fh:
            pickle.dump(kwargs, fh, protocol=4)
    else:
        with open(file, "wb") as fh:
            pickle.dump(kwargs, fh, protocol=4)


def setup_logger(log_name: str) -> logging.Logger:
    """Setup logger with Rich

    Args:
        log_name (str): _description_

    Returns:
        logging.Logger: _description_
    """
    logger = logging.getLogger(log_name)
    if logger.handlers:
        return logger
    logging.basicConfig(level=logging.ERROR, force=True, handlers=[RichHandler()])
    logger.propagate = False
    logger.setLevel(logging.INFO)
    logger.handlers = [RichHandler()]
    return logger


def env_flag(env_var: str, default: bool = False) -> bool:
    """Return the specified environment variable coerced to a bool, as follows:
    - When the variable is unset, or set to the empty string, return `default`.
    - When the variable is set to a truthy value, returns `True`.
      These are the truthy values:
          - 1
          - true, yes, on
    - When the variable is set to the anything else, returns False.
       Example falsy values:
          - 0
          - no
    - Ignore case and leading/trailing whitespace.
    """
    environ_string = os.environ.get(env_var, "").strip().lower()
    if not environ_string:
        return default
    return environ_string in ["1", "true", "yes", "on"]


def download_file(src: str, dst: str, progress: bool = True):
    """Download file from supplied url to destination.

    Args:
        src (str): Source URL path
        dst (str): Destination file path
        progress (bool, optional): Display progress bar. Defaults to True.

    """
    with requests.get(src, stream=True, timeout=3600 * 24) as r:
        r.raise_for_status()
        req_len = int(r.headers.get("Content-length", 0))
        prog_bar = tqdm(total=req_len, unit="iB", unit_scale=True) if progress else None
        with open(dst, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
                if prog_bar:
                    prog_bar.update(len(chunk))
            # END FOR
        # END WITH
    # END WITH
