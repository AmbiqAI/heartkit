import warnings

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import auc, f1_score, roc_curve


def f1(
    y_true: npt.ArrayLike,
    y_prob: npt.ArrayLike,
    multiclass: bool = False,
    threshold: float = None,
):
    """Compute F1 scores

    Args:
        y_true ( npt.ArrayLike): Y true
        y_prob ( npt.ArrayLike): 2D matrix with class probs
        multiclass (bool, optional): If multiclass. Defaults to False.
        threshold (float, optional): Decision threshold for multiclass. Defaults to None.

    Returns:
        npt.ArrayLike: F1 scores
    """
    if y_prob.ndim != 2:
        raise ValueError("y_prob must be a 2d matrix with class probabilities for each sample")
    if y_true.ndim == 1:  # we assume that y_true is sparse (consequently, multiclass=False)
        if multiclass:
            raise ValueError("if y_true cannot be sparse and multiclass at the same time")
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
    thresholds: float | list[float] | None = None,
) -> tuple[float, float]:
    """Compute F max
    source: https://github.com/helme/ecg_ptbxl_benchmarking
    Args:
        y_true (npt.ArrayLike): Y True
        y_prob (npt.ArrayLike): Y probs
        thresholds (float|list[float]|None, optional): Thresholds. Defaults to None.

    Returns:
        tuple[float, float]: F1 and thresholds
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
    labels: list[str],
    save_path: str | None = None,
    **kwargs,
):
    """Generate confusion matrix plot via matplotlib/seaborn

    Args:
        y_true (npt.ArrayLike): True y labels
        y_pred (npt.ArrayLike): Predicted y labels
        labels (list[str]): Label names
        save_path (str | None): Path to save plot. Defaults to None.
    """
    confusion_mtx = tf.math.confusion_matrix(y_true, y_pred)
    plt.figure(figsize=kwargs.get("figsize", (10, 8)))
    sns.heatmap(confusion_mtx, xticklabels=labels, yticklabels=labels, annot=True, fmt="g")
    plt.xlabel("Prediction")
    plt.ylabel("Label")
    if save_path:
        plt.savefig(save_path)
    plt.close()


def roc_auc_plot(
    y_true: npt.ArrayLike,
    y_prob: npt.ArrayLike,
    labels: list[str],
    save_path: str | None = None,
    **kwargs,
):
    """Generate ROC plot via matplotlib/seaborn
    Args:
        y_true (npt.ArrayLike): True y labels
        y_prob (npt.ArrayLike): Predicted y labels
        labels (list[str]): Label names
        save_path (str | None): Path to save plot. Defaults to None.
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


def macro_precision_recall(y_true, y_prob, thresholds):
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


def challenge2020_metrics(y_true, y_pred, beta_f=2, beta_g=2, class_weights=None, single=False):
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
        f_beta += w_k * ((1 + beta_f**2) * tp) / ((1 + beta_f**2) * tp + fp + beta_f**2 * fn)
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
