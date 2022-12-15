import warnings
from typing import Callable, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import auc, f1_score, roc_curve

from .utils import setup_logger

logger = setup_logger(__name__)


def f1(
    y_true: npt.ArrayLike,
    y_prob: npt.ArrayLike,
    multiclass: bool = False,
    threshold: float = None,
):
    """Compute F1 scores

    Args:
        y_true ( npt.ArrayLike): _description_
        y_prob ( npt.ArrayLike): 2D matrix with class probs
        multiclass (bool, optional): If multiclass. Defaults to False.
        threshold (float, optional): Decision threshold for multiclass. Defaults to None.

    Returns:
        _type_: _description_
    """
    if y_prob.ndim != 2:
        raise ValueError(
            "y_prob must be a 2d matrix with class probabilities for each sample"
        )
    if (
        y_true.ndim == 1
    ):  # we assume that y_true is sparse (consequently, multiclass=False)
        if multiclass:
            raise ValueError(
                "if y_true cannot be sparse and multiclass at the same time"
            )
        depth = y_prob.shape[1]
        y_true = _one_hot(y_true, depth)
    if multiclass:
        if threshold is None:
            threshold = 0.5
        y_pred = y_prob >= threshold
    else:
        y_pred = y_prob >= np.max(y_prob, axis=1)[:, None]
    return f1_score(y_true, y_pred, average="macro")


def f_max(
    y_true: npt.ArrayLike,
    y_prob: npt.ArrayLike,
    thresholds: Optional[Union[float, List[float]]] = None,
):
    """Compute F max
    source: https://github.com/helme/ecg_ptbxl_benchmarking
    Args:
        y_true (npt.ArrayLike): _description_
        y_prob (npt.ArrayLike): _description_
        thresholds (_type_, optional): _description_. Defaults to None.

    Returns:
        Tuple[float]: _description_
    """
    if thresholds is None:
        thresholds = np.linspace(0, 1, 100)
    pr, rc = macro_precision_recall(y_true, y_prob, thresholds)
    f1s = (2 * pr * rc) / (pr + rc)
    i = np.nanargmax(f1s)
    return f1s[i], thresholds[i]


def confusion_matrix_plot(
    y_true: npt.ArrayLike,
    y_pred: npt.ArrayLike,
    labels: List[str],
    save_path: Optional[str] = None,
    **kwargs,
):
    """Generate confusion matrix plot via matplotlib/seaborn

    Args:
        y_true (npt.ArrayLike): True y labels
        y_pred (npt.ArrayLike): Predicted y labels
        labels (List[str]): Label names
        save_path (Optional[str]): Path to save plot. Defaults to None.
    """
    confusion_mtx = tf.math.confusion_matrix(y_true, y_pred)
    plt.figure(figsize=kwargs.get("figsize", (10, 8)))
    sns.heatmap(
        confusion_mtx, xticklabels=labels, yticklabels=labels, annot=True, fmt="g"
    )
    plt.xlabel("Prediction")
    plt.ylabel("Label")
    if save_path:
        plt.savefig(save_path)
    plt.close()


def roc_auc_plot(
    y_true: npt.ArrayLike,
    y_prob: npt.ArrayLike,
    labels: List[str],
    save_path: Optional[str] = None,
    **kwargs,
):
    """Generate ROC plot via matplotlib/seaborn
    Args:
        y_true (npt.ArrayLike): True y labels
        y_prob (npt.ArrayLike): Predicted y labels
        labels (List[str]): Label names
        save_path (Optional[str]): Path to save plot. Defaults to None.
    """

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=kwargs.get("figsize", (10, 8)))
    lw = 2
    plt.plot(
        fpr,
        tpr,
        lw=lw,
        color="darkorange",
        label=f"ROC curve (area = {roc_auc:0.2f})",
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC-AUC")
    plt.legend(loc="lower right")
    if save_path:
        plt.savefig(save_path)
    plt.close()


def macro_precision_recall(y_true, y_prob, thresholds):  # multi-class multi-output
    """source: https://github.com/helme/ecg_ptbxl_benchmarking"""
    # expand analysis to the number of thresholds
    y_true = np.repeat(y_true[None, :, :], len(thresholds), axis=0)
    y_prob = np.repeat(y_prob[None, :, :], len(thresholds), axis=0)
    y_pred = y_prob >= thresholds[:, None, None]

    # compute true positives
    tp = np.sum(np.logical_and(y_true, y_pred), axis=2)

    # compute macro average precision handling all warnings
    with np.errstate(divide="ignore", invalid="ignore"):
        den = np.sum(y_pred, axis=2)
        precision = tp / den
        precision[den == 0] = np.nan
        with warnings.catch_warnings():  # for nan slices
            warnings.simplefilter("ignore", category=RuntimeWarning)
            av_precision = np.nanmean(precision, axis=1)

    # compute macro average recall
    recall = tp / np.sum(y_true, axis=2)
    av_recall = np.mean(recall, axis=1)

    return av_precision, av_recall


def challenge2020_metrics(
    y_true, y_pred, beta_f=2, beta_g=2, class_weights=None, single=False
):
    """source: https://github.com/helme/ecg_ptbxl_benchmarking"""
    num_samples, num_classes = y_true.shape
    if single:  # if evaluating single class in case of threshold-optimization
        sample_weights = np.ones(num_samples)
    else:
        sample_weights = y_true.sum(axis=1)
    if class_weights is None:
        class_weights = np.ones(num_classes)
    f_beta = 0
    g_beta = 0
    for k, w_k in enumerate(class_weights):
        tp, fp, tn, fn = 0.0, 0.0, 0.0, 0.0
        for i in range(num_samples):
            if y_true[i, k] == y_pred[i, k] == 1:
                tp += 1.0 / sample_weights[i]
            if y_pred[i, k] == 1 and y_true[i, k] != y_pred[i, k]:
                fp += 1.0 / sample_weights[i]
            if y_true[i, k] == y_pred[i, k] == 0:
                tn += 1.0 / sample_weights[i]
            if y_pred[i, k] == 0 and y_true[i, k] != y_pred[i, k]:
                fn += 1.0 / sample_weights[i]
        f_beta += (
            w_k
            * ((1 + beta_f**2) * tp)
            / ((1 + beta_f**2) * tp + fp + beta_f**2 * fn)
        )
        g_beta += w_k * tp / (tp + fp + beta_g * fn)
    f_beta /= class_weights.sum()
    g_beta /= class_weights.sum()
    return {"F_beta": f_beta, "G_beta": g_beta}


def _one_hot(x: npt.ArrayLike, depth: int) -> npt.ArrayLike:
    """Generate one hot encoding

    Args:
        x (npt.ArrayLike): Categories
        depth (int): Depth

    Returns:
        npt.ArrayLike: One hot encoded
    """
    x_one_hot = np.zeros((x.size, depth))
    x_one_hot[np.arange(x.size), x] = 1
    return x_one_hot


def multi_f1(y_true: npt.ArrayLike, y_prob: npt.ArrayLike):
    """Compute multi-class F1

    Args:
        y_true (npt.ArrayLike): _description_
        y_prob (npt.ArrayLike): _description_

    Returns:
        _type_: _description_
    """
    return f1(y_true, y_prob, multiclass=True, threshold=0.5)


class CustomCheckpoint(tf.keras.callbacks.Callback):
    """Custom keras callback checkpoint"""

    def __init__(
        self,
        filepath: str,
        data: Tuple[npt.ArrayLike, npt.ArrayLike],
        score_fn: Callable[[Tuple[npt.ArrayLike, npt.ArrayLike]], npt.ArrayLike],
        best: float = -np.Inf,
        save_best_only: bool = False,
        batch_size: Optional[bool] = None,
        verbose: int = 0,
    ):
        """Custom keras callback checkpoint

        Args:
            filepath (str): Save checkpoint filepath
            data (Tuple[npt.ArrayLike, npt.ArrayLike]): Data
            score_fn (Callable[[Tuple[npt.ArrayLike, npt.ArrayLike]], npt.ArrayLike]): Scoring function
            best (float, optional): Current best score. Defaults to -np.Inf.
            save_best_only (bool, optional): Save best checkpoint only. Defaults to False.
            batch_size (Optional[bool], optional): Batch size. Defaults to None.
            verbose (int, optional): Verbosity. Defaults to 0.
        """
        super().__init__()
        self.filepath = filepath
        self.data = data
        self.score_fn = score_fn
        self.save_best_only = save_best_only
        self.batch_size = batch_size
        self.verbose = verbose
        self.best = best

    def on_epoch_end(self, epoch: int, logs=None):
        logs = logs or {}
        x, y_true = self.data
        y_prob = self.model.predict(x, batch_size=self.batch_size)
        score = self.score_fn(y_true, y_prob)
        logs.update({self.metric_name: score})
        filepath = self.filepath.format(epoch=epoch + 1, **logs)
        if score > self.best:
            logger.debug(
                (
                    f"\nEpoch {epoch+1:05d}: {self.metric_name} ({score:.05f}) improved from {self.best:0.5f}"
                    f"saving model to {self.filepath}"
                )
            )
            self.model.save_weights(filepath, overwrite=True)
            self.best = score
        elif not self.save_best_only:
            logger.debug(
                (
                    f"\nEpoch {epoch+1:05d}: {self.metric_name} ({score:.05f}) did not improve from {self.best:0.5f}"
                    f"saving model to {self.filepath}"
                )
            )
            self.model.save_weights(filepath, overwrite=True)
        else:
            logger.debug(
                f"\nEpoch {epoch+1:05d}: {self.metric_name} ({score:.05f}) did not improve from {self.best:0.5f}"
            )

    @property
    def metric_name(self) -> str:
        """Get metric name

        Returns:
            str: name
        """
        return self.score_fn.__name__
