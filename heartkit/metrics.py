import os
import warnings
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import seaborn as sns
from sklearn.metrics import auc, confusion_matrix, f1_score, jaccard_score, roc_curve


def compute_iou(
    y_true: npt.NDArray,
    y_pred: npt.NDArray,
    average: Literal["micro", "macro", "weighted"] = "micro",
) -> float:
    """Compute IoU

    Args:
        y_true (npt.NDArray): Y true
        y_pred (npt.NDArray): Y predicted

    Returns:
        float: IoU
    """
    # intersect = np.logical_and(y_true, y_pred)
    # union = np.logical_or(y_true, y_pred)
    # return np.sum(intersect) / np.sum(union)
    return jaccard_score(y_true.flatten(), y_pred.flatten(), average=average)


def f1(
    y_true: npt.NDArray,
    y_prob: npt.NDArray,
    multiclass: bool = False,
    threshold: float = None,
) -> npt.NDArray | float:
    """Compute F1 scores

    Args:
        y_true ( npt.NDArray): Y true
        y_prob ( npt.NDArray): 2D matrix with class probs
        multiclass (bool, optional): If multiclass. Defaults to False.
        threshold (float, optional): Decision threshold for multiclass. Defaults to None.

    Returns:
        npt.NDArray|float: F1 scores
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
    y_true: npt.NDArray,
    y_prob: npt.NDArray,
    thresholds: float | list[float] | None = None,
) -> tuple[float, float]:
    """Compute F max
    source: https://github.com/helme/ecg_ptbxl_benchmarking
    Args:
        y_true (npt.NDArray): Y True
        y_prob (npt.NDArray): Y probs
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
    y_true: npt.NDArray,
    y_pred: npt.NDArray,
    labels: list[str],
    save_path: str | None = None,
    normalize: Literal["true", "pred", "all"] | None = False,
    **kwargs,
) -> tuple[plt.Figure, plt.Axes] | None:
    """Generate confusion matrix plot via matplotlib/seaborn

    Args:
        y_true (npt.NDArray): True y labels
        y_pred (npt.NDArray): Predicted y labels
        labels (list[str]): Label names
        save_path (str | None): Path to save plot. Defaults to None.
    """

    cm = confusion_matrix(y_true, y_pred)
    cmn = cm
    ann = True
    fmt = "g"
    if normalize:
        cmn = confusion_matrix(y_true, y_pred, normalize=normalize)
        ann = np.asarray(
            [f"{c:g}{os.linesep}{nc:.2%}" for c, nc in zip(cm.flatten(), cmn.flatten())]
        ).reshape(cm.shape)
        fmt = ""
    # END IF
    fig, ax = plt.subplots(figsize=kwargs.get("figsize", (10, 8)))
    sns.heatmap(cmn, xticklabels=labels, yticklabels=labels, annot=ann, fmt=fmt, ax=ax)
    ax.set_xlabel("Prediction")
    ax.set_ylabel("Label")
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
        return None
    return fig, ax


def roc_auc_plot(
    y_true: npt.NDArray,
    y_prob: npt.NDArray,
    labels: list[str],
    save_path: str | None = None,
    **kwargs,
):
    """Generate ROC plot via matplotlib/seaborn
    Args:
        y_true (npt.NDArray): True y labels
        y_prob (npt.NDArray): Predicted y labels
        labels (list[str]): Label names
        save_path (str | None): Path to save plot. Defaults to None.
    """

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=kwargs.get("figsize", (10, 8)))
    label = f"ROC curve (area = {roc_auc:0.2f})"
    ax.plot(fpr, tpr, lw=2, color="darkorange", label=label)
    ax.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC-AUC")
    fig.legend(loc="lower right")
    if save_path:
        fig.savefig(save_path)
        plt.close(fig)
        return None
    return fig, ax


def macro_precision_recall(
    y_true: npt.NDArray, y_prob: npt.NDArray, thresholds: npt.NDArray
):
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


def _one_hot(x: npt.NDArray, depth: int) -> npt.NDArray:
    """Generate one hot encoding

    Args:
        x (npt.NDArray): Categories
        depth (int): Depth

    Returns:
        npt.NDArray: One hot encoded
    """
    x_one_hot = np.zeros((x.size, depth))
    x_one_hot[np.arange(x.size), x] = 1
    return x_one_hot


def multi_f1(y_true: npt.NDArray, y_prob: npt.NDArray):
    """Compute multi-class F1

    Args:
        y_true (npt.NDArray): _description_
        y_prob (npt.NDArray): _description_

    Returns:
        _type_: _description_
    """
    return f1(y_true, y_prob, multiclass=True, threshold=0.5)
