import gzip
import hashlib
import logging
import os
import pickle
from pathlib import Path
from string import Template
from typing import Any

import numpy as np
import requests
from rich.logging import RichHandler
from tqdm import tqdm


def setup_logger(log_name: str) -> logging.Logger:
    """Setup logger with Rich

    Args:
        log_name (str): Logger name

    Returns:
        logging.Logger: Logger
    """
    new_logger = logging.getLogger(log_name)
    if new_logger.handlers:
        return new_logger
    logging.basicConfig(level=logging.ERROR, force=True, handlers=[RichHandler()])
    new_logger.propagate = False
    new_logger.setLevel(logging.INFO)
    new_logger.handlers = [RichHandler()]
    return new_logger


logger = setup_logger(__name__)


def set_random_seed(seed: int | None = None) -> int:
    """Set random seed across libraries: Keras, Numpy, Python

    Args:
        seed (int | None, optional): Random seed state to use. Defaults to None.

    Returns:
        int: Random seed
    """
    seed = seed or np.random.randint(2**16)
    try:
        import keras  # pylint: disable=import-outside-toplevel
    except ImportError:
        pass
    else:
        keras.utils.set_random_seed(seed)
    return seed


def load_pkl(file: str, compress: bool = True) -> dict[str, Any]:
    """Load pickled file.

    Args:
        file (str): File path (.pkl)
        compress (bool, optional): If file is compressed. Defaults to True.

    Returns:
        dict[str, Any]: Dictionary of pickled objects
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

    Args:
        env_var (str): Environment variable name
        default (bool, optional): Default value. Defaults to False.

    Returns:
        bool: Value of environment variable
    """
    environ_string = os.environ.get(env_var, "").strip().lower()
    if not environ_string:
        return default
    return environ_string in ["1", "true", "yes", "on"]


def compute_checksum(file: Path, checksum_type: str = "md5", chunk_size: int = 8192) -> str:
    """Compute checksum of file.

    Args:
        file (Path): File path
        checksum_type (str, optional): Checksum type. Defaults to "md5".
        chunk_size (int, optional): Chunk size. Defaults to 8192.

    Returns:
        str: Checksum value
    """
    if not file.is_file():
        raise FileNotFoundError(f"File {file} not found.")
    hash_algo = hashlib.new(checksum_type)
    with open(file, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            hash_algo.update(chunk)
    return hash_algo.hexdigest()


def download_file(
    src: str,
    dst: Path,
    progress: bool = True,
    chunk_size: int = 8192,
    checksum: str | None = None,
    checksum_type: str = "size",
    timeout: int = 3600 * 24,
):
    """Download file from supplied url to destination streaming.

    checksum: hd5, sha256, sha512, size

    Args:
        src (str): Source URL path
        dst (PathLike): Destination file path
        progress (bool, optional): Display progress bar. Defaults to True.
        chunk_size (int, optional): Chunk size. Defaults to 8192.
        checksum (str|None, optional): Checksum value. Defaults to None.
        checksum_type (str|None, optional): Checksum type or size. Defaults to None.

    Raises:
        ValueError: If checksum doesn't match


    """

    # If file exists and checksum matches, skip download
    if dst.is_file() and checksum:
        match checksum_type:
            case "size":
                # Get number of bytes in file
                calculated_checksum = str(dst.stat().st_size)
            case _:
                calculated_checksum = compute_checksum(dst, checksum_type, chunk_size)
        if calculated_checksum == checksum:
            logger.debug(f"File {dst} already exists and checksum matches. Skipping...")
            return
        # END IF
    # END IF

    # Create parent directory if not exists
    dst.parent.mkdir(parents=True, exist_ok=True)

    # Download file in chunks
    with requests.get(src, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        req_len = int(r.headers.get("Content-length", 0))
        prog_bar = tqdm(total=req_len, unit="iB", unit_scale=True) if progress else None
        with open(dst, "wb") as f:
            for chunk in r.iter_content(chunk_size=chunk_size):
                f.write(chunk)
                if prog_bar:
                    prog_bar.update(len(chunk))
            # END FOR
        # END WITH
    # END WITH


def resolve_template_path(fpath: Path, **kwargs: Any) -> Path:
    """Resolve templated path w/ supplied substitutions.

    Args:
        fpath (Path): File path
        **kwargs (Any): Template arguments

    Returns:
        Path: Resolved file path
    """
    return Path(Template(str(fpath)).safe_substitute(**kwargs))
