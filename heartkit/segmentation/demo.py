import numpy as np
import physiokit as pk
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from tqdm import tqdm

from ..defines import HeartDemoParams, HeartSegment
from ..rpc.backends import EvbBackend, PcBackend
from ..utils import setup_logger
from .defines import get_class_mapping, get_class_names, get_classes
from .utils import load_datasets, prepare

logger = setup_logger(__name__)


def demo(params: HeartDemoParams):
    """Run segmentation demo.

    Args:
        params (SKDemoParams): Demo parameters
    """

    params.datasets = getattr(params, "datasets", ["ludb"])
    params.num_pts = getattr(params, "num_pts", 1000)

    bg_color = "rgba(38,42,50,1.0)"
    primary_color = "#11acd5"
    secondary_color = "#ce6cff"
    tertiary_color = "rgb(234,52,36)"
    quaternary_color = "rgb(92,201,154)"
    plotly_template = "plotly_dark"

    # Load backend inference engine
    BackendRunner = EvbBackend if params.backend == "evb" else PcBackend
    runner = BackendRunner(params=params)

    # Load data
    classes = get_classes(params.num_classes)
    class_names = get_class_names(params.num_classes)
    class_map = get_class_mapping(params.num_classes)

    ds = load_datasets(
        ds_path=params.ds_path,
        frame_size=10 * params.sampling_rate,
        sampling_rate=params.sampling_rate,
        class_map=class_map,
        dataset_names=params.datasets,
        num_pts=params.num_pts,
    )[0]
    x = next(ds.signal_generator(ds.uniform_patient_generator(patient_ids=ds.get_test_patient_ids(), repeat=False)))

    # Run inference
    runner.open()
    logger.info("Running inference")
    y_pred = np.zeros(x.size, dtype=np.int32)
    for i in tqdm(range(0, x.size, params.frame_size), desc="Inference"):
        if i + params.frame_size > x.size:
            start, stop = x.size - params.frame_size, x.size
        else:
            start, stop = i, i + params.frame_size
        xx = prepare(x[start:stop], sample_rate=params.sampling_rate, preprocesses=params.preprocesses)
        runner.set_inputs(xx)
        runner.perform_inference()
        yy = runner.get_outputs()
        y_pred[start:stop] = np.argmax(yy, axis=-1).flatten()
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

    # Extract R peaks
    pred_bounds = np.concatenate(([0], np.diff(y_pred).nonzero()[0] + 1, [y_pred.size - 1]))
    peaks = []
    for i in range(1, len(pred_bounds)):
        start, stop = pred_bounds[i - 1], pred_bounds[i]
        duration = 1000 * (stop - start) / params.sampling_rate
        if y_pred[start] == class_map.get(HeartSegment.qrs, -1) and (duration > 20):
            peaks.append(start + np.argmax(np.abs(x[start:stop])))
    peaks = np.array(peaks)

    # Compute R-R intervals
    rri = pk.ecg.compute_rr_intervals(peaks)
    mask = pk.ecg.filter_rr_intervals(rri, sample_rate=params.sampling_rate)
    # Compute heart rate
    hr_bpm = 60 / (np.nanmean(rri[mask == 0]) / params.sampling_rate)
    # Compute HRV metrics
    hrv_td = pk.hrv.compute_hrv_time(rri[mask == 0], sample_rate=params.sampling_rate)
    band_names = ["VLF", "LF", "HF", "VHF"]
    bands = [(0.0033, 0.04), (0.04, 0.15), (0.15, 0.4), (0.4, 0.5)]
    hrv_fd = pk.hrv.compute_hrv_frequency(
        peaks[mask == 0], rri[mask == 0], bands=bands, sample_rate=params.sampling_rate
    )

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

    for peak in peaks:
        fig.add_vline(
            x=ts[peak],
            line_width=1,
            line_dash="dash",
            line_color="white",
            annotation={"text": "R-Peak", "textangle": -90},
            row=1,
            col=1,
            secondary_y=False,
        )
    fig.update_xaxes(title_text="Time (s)", row=1, col=1)
    fig.update_yaxes(title_text="ECG", row=1, col=1)

    for i, label in enumerate(classes):
        if label <= 0:
            continue
        fig.add_trace(
            go.Scatter(
                x=ts,
                y=np.where(y_pred == label, x, np.nan),
                name=class_names[i],
                mode="lines",
                line_width=2,
            ),
            row=1,
            col=1,
            secondary_y=False,
        )
    # END FOR

    fig.add_trace(
        go.Bar(
            x=np.array([b.total_power for b in hrv_fd.bands]) / hrv_fd.total_power,
            y=band_names,
            marker_color=[primary_color, secondary_color, tertiary_color, quaternary_color],
            orientation="h",
            showlegend=False,
        ),
        row=2,
        col=2,
    )
    fig.update_xaxes(title_text="Normalized Power", row=2, col=2)

    rri_ms = 1000 * rri / params.sampling_rate
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

    fig.add_trace(
        go.Table(
            header=dict(
                values=["Heart Rate <br> Variability Metric", "<br> Value"],
                font=dict(size=16, color="white"),
                height=60,
                fill_color=primary_color,
                align=["left"],
            ),
            cells=dict(
                values=[
                    ["Heart Rate", "NN Mean", "NN St. Dev", "SD RMS"],
                    [f"{hr_bpm:.0f} BPM", f"{hrv_td.mean_nn:.1f} ms", f"{hrv_td.sd_nn:.1f} ms", f"{hrv_td.rms_sd:.1f}"],
                ],
                font=dict(size=14),
                height=40,
                fill_color=bg_color,
                align=["left"],
            ),
        ),
        row=2,
        col=3,
    )

    fig.update_layout(
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template=plotly_template,
        height=800,
        plot_bgcolor=bg_color,
        paper_bgcolor=bg_color,
        margin=dict(l=10, r=10, t=80, b=80),
        title="HeartKit: Segmentation Demo",
    )

    fig.write_html(params.job_dir / "demo.html", include_plotlyjs="cdn", full_html=False)
    fig.show()

    logger.info(f"Report saved to {params.job_dir / 'demo.html'}")
