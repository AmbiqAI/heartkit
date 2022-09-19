import os
import sys
import requests
import logging
from logging.handlers import RotatingFileHandler
import gzip
import random
import pickle
from typing import Optional
import tensorflow as tf
import numpy as np
import pandas as pd
from tqdm import tqdm

def set_random_seed(seed: Optional[int] = None):
    seed = seed or np.random.randint(2 ** 16)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    return seed

def xxd_c_dump(src_path: str, dst_path: str, var_name: str = 'g_model', chunk_len: int = 12):
    """ Generate C like char array of hex values from binary source. Equivalent to `xxd -i src_path > dst_path`
        but with added features to provide # columns and variable name.
    Args:
        src_path (str): Binary file source path
        dst_path (str): C file destination path
        var_name (str, optional): C variable name. Defaults to 'g_model'.
        chunk_len (int, optional): # of elements per row. Defaults to 12.
    """
    var_len = 0;
    with open(src_path, 'rb') as rfp, open(dst_path, 'w') as wfp:
        wfp.write(f"const unsigned char {var_name}[] = {{{os.linesep}");
        for chunk in iter(lambda: rfp.read(chunk_len), b''):
            wfp.write("  " + ", ".join((f'0x{c:02x}' for c in chunk)) + f", {os.linesep}")
            var_len += len(chunk)
        # END FOR
        wfp.write(f"}};{os.linesep}");
        wfp.write(f"const unsigned int {var_name}_len = {var_len};{os.linesep}")
    # END WITH


def pad_sequences(x, max_len=None, padding='pre'):
    """
    Pads sequences shorter than `max_len` and trims those longer than `max_len`.
    @param x: Array of sequences.
    @param max_len: Maximal length of the padded sequences. Defaults to the longest sequence.
    @param padding: Type of padding: 'pre' before sequence, 'post' after sequence.
    @return: Array of shape (num_sequences, max_len) containing the padded sequences.
    """
    if max_len is None:
        max_len = max(map(len, x))
    x_shape = x[0].shape
    x_dtype = x[0].dtype
    x_padded = np.zeros((len(x), max_len) + x_shape[1:], dtype=x_dtype)
    for i, x_i in enumerate(x):
        trim_len = min(max_len, len(x_i))
        if padding == 'pre':
            x_padded[i, -trim_len:] = x_i[-trim_len:]
        elif padding == 'post':
            x_padded[i, :trim_len] = x_i[:trim_len]
        else:
            raise ValueError('Unknown padding: %s' % padding)
    return x_padded


def create_predictions_frame(y_prob, y_true=None, y_pred=None, class_names=None, record_ids=None):
    """
    Create predictions matrix.
    @param y_prob: Float array with class probabilities of shape (num_samples,) or (num_samples, num_classes).
    @param y_true: Integer array with true labels of shape (num_samples,) or (num_samples, num_classes).
    @param y_pred: Integer array with class predictions of shape (num_samples,) or (num_samples, num_classes).
    @param class_names: Array of class names of shape (num_classes,).
    @param record_ids: Array of record names of shape (num_samples,).
    @return: DataFrame that contains the predictions matrix.
    """
    y_prob = np.squeeze(y_prob)
    if y_prob.ndim == 1:  # binary classification
        y_prob = np.stack([1 - y_prob, y_prob], axis=1)
    num_classes = y_prob.shape[1]
    if class_names is None:
        # use index of the label as a class name
        class_names = np.arange(num_classes)
    elif len(class_names) != num_classes:
        raise ValueError('length of class_names does not match with the number of classes')
    columns = ['prob_{}'.format(label) for label in class_names]
    data = {column: y_prob[:, i] for i, column in enumerate(columns)}
    if y_pred is not None:
        y_pred = np.squeeze(y_pred)
        if y_pred.ndim == 1:
            y_pred = np.stack([1 - y_pred, y_pred], axis=1)
        if y_pred.shape != y_prob.shape:
            raise ValueError('y_prob and y_pred shapes do not match')
        y_pred_columns = ['pred_{}'.format(label) for label in class_names]
        y_pred_data = {column: y_pred[:, i] for i, column in enumerate(y_pred_columns)}
        columns = columns + y_pred_columns
        data = {**data, **y_pred_data}
    if y_true is not None:
        y_true = np.squeeze(y_true)
        if y_true.ndim == 1:  # class indices
            # search for true labels that do not correspond to any column in the predictions matrix
            unknown_labels = np.setdiff1d(y_true, np.arange(num_classes))
            if len(unknown_labels) > 0:
                raise ValueError('Unknown labels encountered: %s' % unknown_labels)
            y_true = np.eye(num_classes)[y_true]
        if y_true.shape != y_prob.shape:
            raise ValueError('y_prob and y_true shapes do not match')
        y_true_columns = ['true_{}'.format(label) for label in class_names]
        y_true_data = {column: y_true[:, i] for i, column in enumerate(y_true_columns)}
        columns = y_true_columns + columns
        data = {**data, **y_true_data}
    predictions_frame = pd.DataFrame(
        data=data,
        columns=columns)
    if record_ids is not None:
        predictions_frame.insert(0, 'record_name', record_ids)
    return predictions_frame


def read_predictions(file):
    """
    Read predictions matrix.
    @param file: path to the csv file with predictions.
    @return: dictionary with keys: `y_prob`, (optionally) `y_true`, (optionally) `y_pred`, and `classes`.
    """
    df = pd.read_csv(file)
    classes = [label[5:] for label in df.columns if label.startswith('prob')]
    predictions = {}
    for prefix in ['true', 'pred', 'prob']:
        col_names = ['{}_{}'.format(prefix, label) for label in classes]
        col_names = [name for name in col_names if name in df.columns]
        if col_names:
            predictions['y_{}'.format(prefix)] = df[col_names].values
    predictions['classes'] = classes
    return predictions


def matches_spec(o, spec, ignore_batch_dim=False):
    """
    Test whether data object matches the desired spec.
    @param o: Data object.
    @param spec: Metadata for describing the the data object.
    @param ignore_batch_dim: Ignore first dimension when checking the shapes.
    @return: True if the data object matches the spec, otherwise False.
    """
    if isinstance(spec, (list, tuple)):
        if not isinstance(o, (list, tuple)):
            raise ValueError('data object is not a list or tuple which is required by the spec: {}'.format(spec))
        if len(spec) != len(o):
            raise ValueError('data object has a different number of elements than the spec: {}'.format(spec))
        for i in range(len(spec)):
            if not matches_spec(o[i], spec[i], ignore_batch_dim=ignore_batch_dim):
                return False
        return True
    elif isinstance(spec, dict):
        if not isinstance(o, dict):
            raise ValueError('data object is not a dict which is required by the spec: {}'.format(spec))
        if spec.keys() != o.keys():
            raise ValueError('data object has different keys than those specified in the spec: {}'.format(spec))
        for k in spec:
            if not matches_spec(o[k], spec[k], ignore_batch_dim=ignore_batch_dim):
                return False
            return True
    else:
        spec_shape = spec.shape[1:] if ignore_batch_dim else spec.shape
        o_shape = o.shape[1:] if ignore_batch_dim else o.shape
        return spec_shape == o_shape and spec.dtype == o.dtype


def running_mean_std(iterator, dtype=None):
    """
    Calculate mean and standard deviation while iterating over the data iterator.
    @param iterator: Data iterator.
    @param dtype: Type of accumulators.
    @return: Mean, Std.
    """
    sum_x = np.zeros((), dtype=dtype)
    sum_x2 = np.zeros((), dtype=dtype)
    n = 0
    for x in iterator:
        sum_x += np.sum(x, dtype=dtype)
        sum_x2 += np.sum(x ** 2, dtype=dtype)
        n += x.size
    mean = sum_x / n
    std = np.math.sqrt((sum_x2 / n) - (mean ** 2))
    return mean, std


def buffered_generator(generator, buffer_size):
    """
    Buffer the elements yielded by a generator. New elements replace the oldest elements in the buffer.
    The buffer should be accessed in a read-only manner.
    @param generator: Generator object.
    @param buffer_size: Number of elements in the buffer.
    @return: Generator that yields a buffer.
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


def load_pkl(file, compress=True):
    """ Load pickled file. """
    if compress:
        with gzip.open(file, 'rb') as fh:
            return pickle.load(fh)
    else:
        with open(file, 'rb') as fh:
            return pickle.load(fh)


def save_pkl(file, compress=True, **kwargs):
    """ Save dictionary in a pickle file. """
    if compress:
        with gzip.open(file, 'wb') as fh:
            pickle.dump(kwargs, fh, protocol=4)
    else:
        with open(file, 'wb') as fh:
            pickle.dump(kwargs, fh, protocol=4)


def is_multiclass(labels):
    """ Return true if this is a multiclass task otherwise false. """
    return labels.squeeze().ndim == 2 and any(labels.sum(axis=1) != 1)


def rolling_standardize(x: np.ndarray, win_len: int):
    x_roll = np.lib.stride_tricks.sliding_window_view(x, win_len)
    x_roll_std = np.std(x_roll, axis=-1)
    x_roll_mu = np.mean(x_roll, axis=-1)
    x_std = np.concatenate((np.repeat(x_roll_std[0], x.shape[0] - x_roll_std.shape[0]), x_roll_std))
    x_mu = np.concatenate((np.repeat(x_roll_mu[0], x.shape[0] - x_roll_mu.shape[0]), x_roll_mu))
    x_norm = (x - x_mu)/x_std
    return x_norm

class MaxLevelFilter(logging.Filter):
    def __init__(self, max_level):
        self.max_level = max_level

    def filter(self, log_record):
        return log_record.levelno <= self.max_level

def setup_logger(log_name: str, log_folder: str) -> logging.Logger:
    os.makedirs(log_folder, exist_ok=True)
    log_formatter = logging.Formatter('[%(levelname)s:%(asctime)s] %(message)s', '%Y-%m-%dT%H:%M:%S')
    logger = logging.getLogger(log_name)
    if logger.handlers and len(logger.handlers) > 0:
        return logger
    logger.setLevel(logging.INFO)
    info_file_handler = RotatingFileHandler(
        filename=os.path.join(log_folder, log_name + '.info.log'),
        maxBytes=3 * 1024 * 1024,
        backupCount=10
    )
    info_file_handler.setFormatter(log_formatter)
    info_file_handler.setLevel(logging.INFO)
    info_file_handler.addFilter(MaxLevelFilter(logging.WARNING))
    error_file_handler = RotatingFileHandler(
        filename=os.path.join(log_folder, log_name + '.error.log'),
        maxBytes=3 * 1024 * 1024,
        backupCount=10
    )
    error_file_handler.setFormatter(log_formatter)
    error_file_handler.setLevel(logging.ERROR)
    logger.addHandler(info_file_handler)
    logger.addHandler(error_file_handler)
    # For development print to stdout and stderr
    if os.getenv('PYTHON_ENV') == 'development' or os.getenv('LOG_VERBOSE'):
        so_handler = logging.StreamHandler(sys.stdout)
        so_handler.setFormatter(log_formatter)
        so_handler.setLevel(logging.DEBUG)
        so_handler.addFilter(MaxLevelFilter(logging.WARNING))
        se_handler = logging.StreamHandler(sys.stderr)
        se_handler.setFormatter(log_formatter)
        se_handler.setLevel(logging.WARNING)
        logger.addHandler(so_handler)
        logger.addHandler(se_handler)
        logger.setLevel(logging.DEBUG)
    return logger

def env_flag(env_var: str, default: bool = False) -> bool:
    """
    Return the specified environment variable coerced to a bool, as follows:
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
    with requests.get(src, stream=True) as r:
        r.raise_for_status()
        req_len = int(r.headers.get('Content-length', 0))
        prog_bar = tqdm(total=req_len, unit='iB', unit_scale=True) if progress else None
        with open(dst, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
                if prog_bar:
                    prog_bar.update(len(chunk))
    return dst
