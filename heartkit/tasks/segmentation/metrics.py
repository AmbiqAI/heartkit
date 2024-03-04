import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from .defines import HKSegment


def plot_segmentations(
    data: npt.NDArray,
    seg_mask: npt.NDArray | None = None,
    fig: plt.Figure | None = None,
    ax: plt.Axes | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Generate line plot of ECG data with lines colored based on segmentation mask

    Args:
        data (npt.NDArray): ECG data
        seg_mask (npt.NDArray | None, optional): Segmentation mask. Defaults to None.
        fig (plt.Figure | None, optional): Existing figure handle. Defaults to None.
        ax (plt.Axes | None, optional): Existing axes handle. Defaults to None.

    Returns:
        tuple[plt.Figure, plt.Axes]: Figure and axes handle
    """
    color_map = {
        HKSegment.normal: "lightgray",
        HKSegment.pwave: "blue",
        HKSegment.qrs: "orange",
        HKSegment.twave: "green",
    }
    t = np.arange(0, data.shape[0])
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(10, 4), layout="constrained")
    ax.plot(t, data, color="lightgray")
    if seg_mask is not None:
        pred_bnds = np.where(np.abs(np.diff(seg_mask)) > 0)[0]
        pred_bnds = np.concatenate(([0], pred_bnds, [len(seg_mask) - 1]))
        for i in range(pred_bnds.shape[0] - 1):
            c = color_map.get(seg_mask[pred_bnds[i] + 1], "black")
            ax.plot(
                t[pred_bnds[i] : pred_bnds[i + 1]],
                data[pred_bnds[i] : pred_bnds[i + 1]],
                color=c,
            )
    return fig, ax
