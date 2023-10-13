import numpy as np
import numpy.typing as npt
import pandas as pd


def make_divisible(v: int, divisor: int = 4, min_value: int | None = None) -> int:
    """Ensure layer has # channels divisble by divisor
       https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    Args:
        v (int): # channels
        divisor (int, optional): Divisor. Defaults to 4.
        min_value (int | None, optional): Min # channels. Defaults to None.

    Returns:
        int: # channels
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def get_predicted_threshold_indices(
    y_prob: npt.NDArray,
    y_pred: npt.NDArray,
    threshold: float = 0.5,
) -> npt.NDArray:
    """Get prediction indices that are above threshold (confidence level).
    This is useful to remove weak predictions that can happen due to noisy data or poor model performance.

    Args:
        y_prob (npt.NDArray): Model output as probabilities
        y_pred (npt.NDArray, optional): Model predictions. Defaults to None.
        threshold (float): Confidence level

    Returns:
        npt.NDArray: Indices of y_prob that satisfy threshold
    """

    y_pred_prob = np.take_along_axis(y_prob, np.expand_dims(y_pred, axis=-1), axis=-1).squeeze(axis=-1)
    y_thresh_idx = np.where(y_pred_prob > threshold)[0]
    return y_thresh_idx


def threshold_predictions(
    y_prob: npt.NDArray,
    y_pred: npt.NDArray,
    y_true: npt.NDArray,
    threshold: float = 0.5,
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
    """Get prediction indices that are above threshold (confidence level).
    This is useful to remove weak predictions that can happen due to noisy data or poor model performance.

    Args:
        y_prob (npt.NDArray): Model output as probabilities
        y_pred (npt.NDArray, optional): Model predictions. Defaults to None.
        y_true (npt.NDArray): True labels
        threshold (float): Confidence level. Defaults to 0.5.

    Returns:
        tuple[npt.NDArray, npt.NDArray, npt.NDArray]: Thresholded predictions
    """
    y_thresh_idx = get_predicted_threshold_indices(y_prob, y_pred, threshold)
    y_prob = y_prob[y_thresh_idx]
    y_pred = y_pred[y_thresh_idx]
    y_true = y_true[y_thresh_idx]
    return y_prob, y_pred, y_true


def create_predictions_frame(
    y_prob: npt.NDArray,
    y_true: npt.NDArray | None = None,
    y_pred: npt.NDArray | None = None,
    class_names: list[str] | None = None,
    record_ids: list[str] | None = None,
):
    """Create predictions matrix.
    Args:
        y_prob (npt.NDArray): Array of class probabilities of shape (num_samples,) or (num_samples, num_classes).
        y_true (npt.NDArray | None): Integer array with true labels of shape (num_samples,) or (num_samples, num_classes).
        y_pred (npt.NDArray | None): Integer array with class predictions of shape (num_samples,) or (num_samples, num_classes).
        class_names (list[str] | None): Array of class names of shape (num_classes,).
        record_ids (list[str] | None): Array of record names of shape (num_samples,).
    Returns:
        pd.DataFrame: Predictions matrix.
    """
    y_prob = np.squeeze(y_prob)
    if y_prob.ndim == 1:  # binary classification
        y_prob = np.stack([1 - y_prob, y_prob], axis=1)
    num_classes = y_prob.shape[1]
    if class_names is None:
        # use index of the label as a class name
        class_names = np.arange(num_classes)
    elif len(class_names) != num_classes:
        raise ValueError("length of class_names does not match with the number of classes")
    columns = [f"prob_{label}" for label in class_names]
    data = {column: y_prob[:, i] for i, column in enumerate(columns)}
    if y_pred is not None:
        y_pred = np.squeeze(y_pred)
        if y_pred.ndim == 1:
            y_pred = np.stack([1 - y_pred, y_pred], axis=1)
        if y_pred.shape != y_prob.shape:
            raise ValueError("y_prob and y_pred shapes do not match")
        y_pred_columns = [f"pred_{label}" for label in class_names]
        y_pred_data = {column: y_pred[:, i] for i, column in enumerate(y_pred_columns)}
        columns = columns + y_pred_columns
        data = {**data, **y_pred_data}
    if y_true is not None:
        y_true = np.squeeze(y_true)
        if y_true.ndim == 1:  # class indices
            # search for true labels that do not correspond to any column in the predictions matrix
            unknown_labels = np.setdiff1d(y_true, np.arange(num_classes))
            if len(unknown_labels) > 0:
                raise ValueError(f"Unknown labels encountered: {unknown_labels}")
            y_true = np.eye(num_classes)[y_true]
        if y_true.shape != y_prob.shape:
            raise ValueError("y_prob and y_true shapes do not match")
        y_true_columns = [f"true_{label}" for label in class_names]
        y_true_data = {column: y_true[:, i] for i, column in enumerate(y_true_columns)}
        columns = y_true_columns + columns
        data = {**data, **y_true_data}
    predictions_frame = pd.DataFrame(data=data, columns=columns)
    if record_ids is not None:
        predictions_frame.insert(0, "record_name", record_ids)
    return predictions_frame


def read_predictions(file: str):
    """Read predictions matrix.
    Args:
    file (str): Path to the csv file with predictions.
    Returns:
        Dict[str, any]: Keys: `y_prob`, (optionally) `y_true`, (optionally) `y_pred`, and `classes`.
    """
    df = pd.read_csv(file)
    classes = [label[5:] for label in df.columns if label.startswith("prob")]
    predictions = {}
    for prefix in ["true", "pred", "prob"]:
        col_names = [f"{prefix}_{label}" for label in classes]
        col_names = [name for name in col_names if name in df.columns]
        if col_names:
            predictions[f"y_{prefix}"] = df[col_names].values
    predictions["classes"] = classes
    return predictions


def is_multiclass(labels: npt.NDArray) -> bool:
    """Return true if this is a multiclass task otherwise false.

    Args:
        labels (npt.NDArray): List of labels

    Returns:
        bool: If multiclass
    """
    return labels.squeeze().ndim == 2 and any(labels.sum(axis=1) != 1)


def matches_spec(o, spec, ignore_batch_dim: bool = False):
    """Test whether data object matches the desired spec.
    Args:
        o: Data object.
        spec: Metadata for describing the the data object.
        ignore_batch_dim: Ignore first dimension when checking shapes.
    Returns:
        bool: If matches
    """
    if isinstance(spec, (list, tuple)):
        if not isinstance(o, (list, tuple)):
            raise ValueError(f"data object is not a list or tuple which is required by the spec: {spec}")
        if len(spec) != len(o):
            raise ValueError(f"data object has a different number of elements than the spec: {spec}")
        for i, ispec in enumerate(spec):
            if not matches_spec(o[i], ispec, ignore_batch_dim=ignore_batch_dim):
                return False
        return True

    if isinstance(spec, dict):
        if not isinstance(o, dict):
            raise ValueError(f"data object is not a dict which is required by the spec: {spec}")
        if spec.keys() != o.keys():
            raise ValueError(f"data object has different keys than those specified in the spec: {spec}")
        for k in spec:
            if not matches_spec(o[k], spec[k], ignore_batch_dim=ignore_batch_dim):
                return False
            return True

    spec_shape = spec.shape[1:] if ignore_batch_dim else spec.shape
    o_shape = o.shape[1:] if ignore_batch_dim else o.shape
    return spec_shape == o_shape and spec.dtype == o.dtype
