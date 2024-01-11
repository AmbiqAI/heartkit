import numpy as np
import physiokit as pk
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from tqdm import tqdm

from ..defines import HeartDemoParams
from ..rpc.backends import EvbBackend, PcBackend
from ..utils import setup_logger
from .defines import get_class_mapping, get_class_names
from .utils import load_dataset, prepare

logger = setup_logger(__name__)


def demo(params: HeartDemoParams):
    """Run beat classification demo.

    Args:
        params (SKDemoParams): Demo parameters
    """

    bg_color = "rgba(38,42,50,1.0)"
    primary_color = "#11acd5"
    secondary_color = "#ce6cff"
    plotly_template = "plotly_dark"

    # Load backend inference engine
    BackendRunner = EvbBackend if params.backend == "evb" else PcBackend
    runner = BackendRunner(params=params)

    # Load data
    class_names = get_class_names(params.num_classes)
    class_map = get_class_mapping(params.num_classes)

    ds = load_dataset(
        ds_path=params.ds_path,
        frame_size=20 * params.sampling_rate,
        sampling_rate=params.sampling_rate,
        class_map=class_map,
    )
    x = next(
        ds.signal_generator(ds.uniform_patient_generator(patient_ids=ds.get_test_patient_ids(), repeat=False))
    ).flatten()

    # Need to locate peaks, compute
    xf = pk.ecg.clean(x, sample_rate=params.sampling_rate)
    xf = pk.signal.normalize_signal(xf, eps=0.1, axis=None)
    peaks = pk.ecg.find_peaks(xf, sample_rate=params.sampling_rate)
    rri = pk.ecg.compute_rr_intervals(peaks)
    mask = pk.ecg.filter_rr_intervals(rri, sample_rate=params.sampling_rate)

    avg_rr = int(np.mean(rri[mask == 0]))
    print(f"Found {len(peaks)} peaks, average RR interval is {avg_rr} samples, masked {mask.sum()} samples")

    # Run inference
    runner.open()
    logger.info("Running inference")
    y_pred = np.zeros_like(peaks, dtype=np.int32)

    for i in tqdm(range(1, len(rri) - 1), desc="Inference"):
        start = peaks[i] - int(params.frame_size * 0.5)
        stop = start + params.frame_size
        if start - avg_rr < 0 or stop + avg_rr > x.size:
            continue
        xx = np.vstack(
            (
                x[start - avg_rr : stop - avg_rr],
                x[start:stop],
                x[start + avg_rr : stop + avg_rr],
            )
        ).T
        xx = prepare(xx, sample_rate=params.sampling_rate, preprocesses=params.preprocesses)
        runner.set_inputs(xx)
        runner.perform_inference()
        yy = runner.get_outputs()
        y_pred[i] = np.argmax(yy, axis=-1).flatten()
    # END FOR
    runner.close()

    print(y_pred)
    print(mask)

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
        subplot_titles=("ECG Plot", "IBI Poincaré Plot", "HRV Frequency Bands"),
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
    fig.show()

    logger.info(f"Report saved to {params.job_dir / 'demo.html'}")
