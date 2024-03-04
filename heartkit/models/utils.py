import numpy as np
import numpy.typing as npt


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


def is_multiclass(labels: npt.NDArray) -> bool:
    """Return true if this is a multiclass task otherwise false.

    Args:
        labels (npt.NDArray): List of labels

    Returns:
        bool: If multiclass
    """
    return labels.squeeze().ndim == 2 and any(labels.sum(axis=1) != 1)


SpecType = list[npt.ArrayLike] | tuple[npt.ArrayLike] | dict[str, npt.ArrayLike] | npt.ArrayLike


def matches_spec(o: SpecType, spec: SpecType, ignore_batch_dim: bool = False) -> bool:
    """Test whether data object matches the desired spec.

    Args:
        o (SpecType): Data object.
        spec (SpecType): Metadata for describing the the data object.
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
