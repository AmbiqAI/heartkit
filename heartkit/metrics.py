import os
import warnings
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
import plotly.express as px
import seaborn as sns
from sklearn.metrics import (
    auc,
    confusion_matrix,
    f1_score,
    jaccard_score,
    multilabel_confusion_matrix,
    roc_curve,
)


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


def multilabel_confusion_matrix_plot(
    y_true: npt.NDArray,
    y_pred: npt.NDArray,
    labels: list[str],
    save_path: os.PathLike | None = None,
    normalize: Literal["true", "pred", "all"] | None = False,
    max_cols: int = 5,
    **kwargs,
) -> tuple[plt.Figure, plt.Axes] | None:
    """Generate multilabel confusion matrix plot via matplotlib/seaborn

    Args:
        y_true (npt.NDArray): True y labels
        y_pred (npt.NDArray): Predicted y labels
        labels (list[str]): Label names
        save_path (str | None): Path to save plot. Defaults to None.
        normalize (Literal["true", "pred", "all"] | None): Normalize. Defaults to False.
        max_cols (int): Max columns. Defaults to 5.

    Returns:
        tuple[plt.Figure, plt.Axes] | None: Figure and axes
    """
    cms = multilabel_confusion_matrix(y_true, y_pred)
    ncols = min(len(labels), max_cols)
    nrows = len(labels) // ncols + (len(labels) % ncols > 0)
    width = 10
    hdim = width / ncols
    fig, ax = plt.subplots(figsize=kwargs.get("figsize", (width, hdim * nrows)), nrows=nrows, ncols=ncols)
    for i, label in enumerate(labels):
        r = i // ncols
        c = i % ncols
        ann, fmt = True, "g"
        cm = cms[i]  # cm will be 2x2
        cmn = cm
        if normalize == "true":
            cmn = cmn / cmn.sum(axis=1, keepdims=True)
        elif normalize == "pred":
            cmn = cmn / cmn.sum(axis=0, keepdims=True)
        elif normalize == "all":
            cmn = cmn / cmn.sum()
        cmn = np.nan_to_num(cmn)
        if normalize:
            ann = np.asarray([f"{c:g}{os.linesep}{nc:.2%}" for c, nc in zip(cm.flatten(), cmn.flatten())]).reshape(
                cm.shape
            )
            fmt = ""
        # END IF
        cm_ax = ax[c] if nrows <= 1 else ax[r][c]
        sns.heatmap(cmn, xticklabels=False, yticklabels=False, annot=ann, fmt=fmt, ax=cm_ax)
        cm_ax.set_title(label)
    # END FOR
    # Remove unused subplots
    for i in range(len(labels), nrows * ncols):
        r = i // ncols
        c = i % ncols
        cm_ax = ax[c] if nrows <= 1 else ax[r][c]
        fig.delaxes(cm_ax)
    # END FOR
    fig.text(0.5, 0.04, "Prediction", ha="center")
    fig.text(0, 0.5, "Label", va="center", rotation="vertical")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
        return None
    # END IF
    return fig, ax


def confusion_matrix_plot(
    y_true: npt.NDArray,
    y_pred: npt.NDArray,
    labels: list[str],
    save_path: os.PathLike | None = None,
    normalize: Literal["true", "pred", "all"] | None = False,
    **kwargs,
) -> tuple[plt.Figure, plt.Axes] | None:
    """Generate confusion matrix plot via matplotlib/seaborn

    Args:
        y_true (npt.NDArray): True y labels
        y_pred (npt.NDArray): Predicted y labels
        labels (list[str]): Label names
        save_path (str | None): Path to save plot. Defaults to None.

    Returns:
        tuple[plt.Figure, plt.Axes] | None: Figure and axes
    """

    cm = confusion_matrix(y_true, y_pred)
    cmn = cm
    ann = True
    fmt = "g"
    if normalize:
        cmn = confusion_matrix(y_true, y_pred, normalize=normalize)
        ann = np.asarray([f"{c:g}{os.linesep}{nc:.2%}" for c, nc in zip(cm.flatten(), cmn.flatten())]).reshape(cm.shape)
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
    # END IF
    return fig, ax


def px_plot_confusion_matrix(
    y_true: npt.NDArray,
    y_pred: npt.NDArray,
    labels: list[str],
    normalize: Literal["true", "pred", "all"] | None = False,
    save_path: os.PathLike | None = None,
    title: str | None = None,
    width: int | None = None,
    height: int | None = 400,
    bg_color: str = "rgba(38,42,50,1.0)",
):
    """Generate confusion matrix plot via plotly

    Args:
        y_true (npt.NDArray): True y labels
        y_pred (npt.NDArray): Predicted y labels
        labels (list[str]): Label names
        normalize (Literal["true", "pred", "all"] | None): Normalize. Defaults to False.
        save_path (os.PathLike | None): Path to save plot. Defaults to None.
        title (str | None): Title. Defaults to None.
        width (int | None): Width. Defaults to None.
        height (int | None): Height. Defaults to 400.
        bg_color (str): Background color. Defaults to "rgba(38,42,50,1.0)".

    Returns:
        plotly.graph_objs.Figure: Plotly figure
    """

    cm = confusion_matrix(y_true, y_pred)
    cmn = cm
    ann = None
    if normalize:
        cmn = confusion_matrix(y_true, y_pred, normalize=normalize)
        ann = np.asarray([f"{c:g}<br>{nc:.2%}" for c, nc in zip(cm.flatten(), cmn.flatten())]).reshape(cm.shape)

    cmn = pd.DataFrame(cmn, index=labels, columns=labels)
    fig = px.imshow(
        cmn,
        labels=dict(x="Predicted", y="Actual", color="Count", text_auto=True),
        title=title,
        template="plotly_dark",
        color_continuous_scale="Plotly3",
    )
    if ann is not None:
        fig.update_traces(text=ann, texttemplate="%{text}")

    fig.update_layout(
        plot_bgcolor=bg_color,
        paper_bgcolor=bg_color,
        margin=dict(l=20, r=5, t=40, b=20),
        height=height,
        width=width,
        title=title,
    )
    if save_path is not None:
        fig.write_html(save_path, include_plotlyjs="cdn", full_html=False)

    return fig


def roc_auc_plot(
    y_true: npt.NDArray,
    y_prob: npt.NDArray,
    labels: list[str],
    save_path: os.PathLike | None = None,
    **kwargs,
) -> tuple[plt.Figure, plt.Axes] | None:
    """Generate ROC plot via matplotlib/seaborn

    Args:
        y_true (npt.NDArray): True y labels
        y_prob (npt.NDArray): Predicted y labels
        labels (list[str]): Label names
        save_path (str | None): Path to save plot. Defaults to None.

    Returns:
        tuple[plt.Figure, plt.Axes] | None: Figure and axes
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
) -> tuple[np.float_, np.float_]:
    """Compute macro precision and recall
    source: https://github.com/helme/ecg_ptbxl_benchmarking

    Args:
        y_true (npt.NDArray): True y labels
        y_prob (npt.NDArray): Predicted y labels
        thresholds (npt.NDArray): Thresholds

    Returns:
       tuple[np.float_, np.float_]: Precision and recall
    """
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


def multi_f1(y_true: npt.NDArray, y_prob: npt.NDArray) -> npt.NDArray | float:
    """Compute multi-class F1

    Args:
        y_true (npt.NDArray): True y labels
        y_prob (npt.NDArray): Predicted y labels

    Returns:
        npt.NDArray|float: F1 score
    """
    return f1(y_true, y_prob, multiclass=True, threshold=0.5)
