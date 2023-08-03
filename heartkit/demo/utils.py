import logging

import numpy as np
import numpy.typing as npt
import plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from rich.logging import RichHandler

from .defines import HeartBeat, HeartSegment, HeartTask, HKResult, get_class_names

try:
    import orjson  # pylint: disable=import-outside-toplevel,unused-import

    plotly.io.json.config.default_engine = "orjson"
except ImportError:
    pass


def ecg_segmentation_plot(
    data: npt.NDArray[np.float32],
    seg_mask: npt.NDArray[np.uint8],
    fig: go.Figure | None = None,
) -> go.Figure:
    """Generate plotly-based ECG segmentation plot from mask

    Args:
        data (npt.NDArray[np.float32]): ECG data
        seg_mask (npt.NDArray[np.uint8]): Segmentation mask
        fig (go.Figure|None, optional): Plotly figure. Defaults to None.

    Returns:
        go.Figure: Plotly figure
    """
    if fig is None:
        fig = make_subplots(rows=1, cols=1)
    num_pts = data.shape[0]
    if num_pts > 0:
        t = np.arange(0, num_pts)

        # Extract segments from mask
        mask = seg_mask & 0x0F
        pwave = np.where(mask == HeartSegment.pwave, data, np.NAN)
        qrs = np.where(mask == HeartSegment.qrs, data, np.NAN)
        twave = np.where(mask == HeartSegment.twave, data, np.NAN)

        fig.add_trace(go.Scatter(x=t, y=data, line_width=4, name="ECG"), row=1, col=1)
        fig.add_trace(go.Scatter(x=t, y=pwave, line_width=4, name="P wave"), row=1, col=1)
        fig.add_trace(go.Scatter(x=t, y=qrs, line_width=4, name="QRS"), row=1, col=1)
        fig.add_trace(go.Scatter(x=t, y=twave, line_width=4, name="T Wave"), row=1, col=1)

        # Extreact beats (PAC, PVC)
        beats = seg_mask >> 4
        beat_idxs = np.where(beats > 0)[0]
        for i in beat_idxs:
            label = "PAC" if beats[i] == HeartBeat.pac else "PVC"
            fig.add_vline(
                x=i,
                line_color="white",
                annotation_text=label,
                annotation_font_color="white",
                annotation_textangle=90,
                row=1,
                col=1,
            )
    fig.update_xaxes(showgrid=False, visible=False, row=1, col=1)
    fig.update_yaxes(showgrid=False, visible=False, row=1, col=1)
    fig.update_layout(
        template="plotly_dark",
        legend_orientation="h",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0.1)",
        margin=dict(l=0, r=0, t=50, b=20),
    )
    return fig


def hkresult_to_str(result: HKResult) -> str:
    """Format HKResult into string for printing"""
    rhythym_names = get_class_names(HeartTask.hrv)
    num_beats = result.num_norm_beats + result.num_pac_beats + result.num_pvc_beats + result.num_noise_beats
    rhythm = "ARRHYTHMIA" if result.arrhythmia else rhythym_names[result.heart_rhythm]
    return (
        "--------------------------\n"
        "**** HeartKit Results ****\n"
        "--------------------------\n"
        f"  Heart Rate: {result.heart_rate}\n"
        f"Heart Rhythm: {rhythm}\n"
        f" Total Beats: {num_beats}\n"
        f"Normal Beats: {result.num_norm_beats}\n"
        f"   PAC Beats: {result.num_pac_beats}\n"
        f"   PVC Beats: {result.num_pvc_beats}\n"
        f" Noise Beats: {result.num_noise_beats}\n"
        f"  Arrhythmia: {'Detected' if result.arrhythmia else 'Not Detected'}\n"
    )


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
