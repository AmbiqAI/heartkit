import random

import numpy as np
import physiokit as pk
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from rich.console import Console
from tqdm import tqdm

from ...datasets.utils import uniform_id_generator
from ...defines import HKDemoParams
from ...rpc import BackendFactory
from ...utils import setup_logger
from ..utils import load_datasets
from .datasets import preprocess

console = Console()
logger = setup_logger(__name__)


def demo(params: HKDemoParams):
    """Run demo on model.

    Args:
        params (HKDemoParams): Demo parameters
    """

    bg_color = "rgba(38,42,50,1.0)"
    primary_color = "#11acd5"
    secondary_color = "#ce6cff"
    plotly_template = "plotly_dark"

    params.demo_size = params.demo_size or 20 * params.sampling_rate

    # Load backend inference engine
    runner = BackendFactory.create(params.backend, params=params)

    # Load data
    # classes = sorted(list(set(params.class_map.values())))
    class_names = params.class_names or [f"Class {i}" for i in range(params.num_classes)]

    feat_shape = (params.frame_size, 1)
    # class_shape = (params.num_classes,)

    # ds_spec = (
    #     tf.TensorSpec(shape=feat_shape, dtype=tf.float32),
    #     tf.TensorSpec(shape=class_shape, dtype=tf.int32),
    # )

    dsets = load_datasets(datasets=params.datasets)
    ds = random.choice(dsets)

    ds_gen = ds.signal_generator(
        patient_generator=uniform_id_generator(ds.get_test_patient_ids(), repeat=False),
        frame_size=params.demo_size,
        samples_per_patient=5,
        target_rate=params.sampling_rate,
    )
    x = next(ds_gen)

    # Need to locate peaks, compute
    xf = pk.ecg.clean(x, sample_rate=params.sampling_rate)
    xf = pk.signal.normalize_signal(xf, eps=0.1, axis=None)
    peaks = pk.ecg.find_peaks(xf, sample_rate=params.sampling_rate)
    rri = pk.ecg.compute_rr_intervals(peaks)
    mask = pk.ecg.filter_rr_intervals(rri, sample_rate=params.sampling_rate)

    avg_rr = int(np.mean(rri[mask == 0]))

    # Run inference
    runner.open()
    logger.info("Running inference")
    y_pred = np.zeros_like(peaks, dtype=np.int32)

    for i in tqdm(range(1, len(rri) - 1), desc="Inference"):
        start = peaks[i] - int(params.frame_size * 0.5)
        stop = start + params.frame_size
        if start - avg_rr < 0 or stop + avg_rr > x.size:
            continue
        xx = x[start:stop]
        xx = preprocess(x[start:stop], sample_rate=params.sampling_rate, preprocesses=params.preprocesses)
        xx = xx.reshape(feat_shape)
        runner.set_inputs(xx)
        runner.perform_inference()
        yy = runner.get_outputs()
        y_pred[i] = np.argmax(yy, axis=-1).flatten()
    # END FOR
    runner.close()

    # Report
    logger.info("Generating report")

    ts = np.arange(0, x.size) / params.sampling_rate

    fig = make_subplots(
        rows=2,
        cols=3,
        specs=[
            [{"colspan": 3, "type": "xy", "secondary_y": True}, None, None],
            [{"type": "xy"}, {"type": "bar"}, {"type": "table"}],
        ],
        subplot_titles=("ECG Plot", "IBI Poincare Plot", "HRV Frequency Bands"),
        horizontal_spacing=0.1,
        vertical_spacing=0.2,
    )

    # Plot ECG
    fig.add_trace(
        go.Scatter(
            x=ts,
            y=x,
            name="ECG",
            mode="lines",
            line=dict(color=primary_color, width=2),
        ),
        row=1,
        col=1,
        secondary_y=False,
    )

    # White vlines for normal QRS, color for PAC/PVC with labels
    for i, peak in enumerate(peaks):
        color = "white" if y_pred[i] == 0 else secondary_color
        label = class_names[y_pred[i]]
        fig.add_vline(
            x=ts[peak],
            line_width=1,
            line_dash="dash",
            line_color=color,
            annotation={"text": label, "textangle": -90},
            row=1,
            col=1,
            secondary_y=False,
        )
    fig.update_xaxes(title_text="Time (s)", row=1, col=1)
    fig.update_yaxes(title_text="ECG", row=1, col=1)

    # Plot normal beats overlayed

    # Plot poincare plot of RR intervals with color for PAC/PVC
    rri_ms = 1000 * rri[1:-1] / params.sampling_rate
    fig.add_trace(
        go.Scatter(
            x=rri_ms[:-1],
            y=rri_ms[1:],
            mode="markers",
            marker_size=10,
            showlegend=False,
            marker_color=secondary_color,
        ),
        row=2,
        col=1,
        secondary_y=False,
    )
    rr_min, rr_max = np.nanmin(rri_ms) - 20, np.nanmax(rri_ms) + 20
    fig.add_shape(
        type="line",
        layer="below",
        y0=rr_min,
        y1=rr_max,
        x0=rr_min,
        x1=rr_max,
        line=dict(color="white", width=2),
        row=2,
        col=1,
    )
    fig.update_xaxes(title_text="RRn (ms)", range=[rr_min, rr_max], row=2, col=1)
    fig.update_yaxes(title_text="RRn+1 (ms)", range=[rr_min, rr_max], row=2, col=1)

    fig.update_layout(
        template=plotly_template,
        height=800,
        plot_bgcolor=bg_color,
        paper_bgcolor=bg_color,
        margin=dict(l=10, r=10, t=80, b=80),
        legend=dict(groupclick="toggleitem"),
        title="HeartKit: Beat Demo",
    )

    fig.write_html(params.job_dir / "demo.html", include_plotlyjs="cdn", full_html=True)
    logger.info(f"Report saved to {params.job_dir / 'demo.html'}")

    if params.display_report:
        fig.show()
