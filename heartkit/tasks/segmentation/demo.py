import random

import numpy as np
import physiokit as pk
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from tqdm import tqdm

from ...datasets.utils import uniform_id_generator
from ...defines import HKDemoParams
from ...rpc import BackendFactory
from ...utils import setup_logger
from ..utils import load_datasets
from .datasets import augment, preprocess
from .defines import HKSegment


def demo(params: HKDemoParams):
    """Run segmentation demo.

    Args:
        params (HKDemoParams): Demo parameters
    """
    logger = setup_logger(__name__, level=params.verbose)

    bg_color = "rgba(38,42,50,1.0)"
    primary_color = "#11acd5"
    secondary_color = "#ce6cff"
    tertiary_color = "rgb(234,52,36)"
    quaternary_color = "rgb(92,201,154)"
    plotly_template = "plotly_dark"

    signal_type = getattr(params, "signal_type", "ECG")  # ECG or PPG

    params.demo_size = params.demo_size or params.frame_size

    # Load backend inference engine
    runner = BackendFactory.create(params.backend, params=params)

    classes = sorted(list(set(params.class_map.values())))
    class_names = params.class_names or [f"Class {i}" for i in range(params.num_classes)]

    feat_shape = (params.frame_size, 1)
    class_shape = (params.frame_size, params.num_classes)

    # ds_spec = (
    #     tf.TensorSpec(shape=feat_shape, dtype=tf.float32),
    #     tf.TensorSpec(shape=class_shape, dtype=tf.int32),
    # )

    datasets = load_datasets(datasets=params.datasets)
    ds = random.choice(datasets)

    ds_gen = ds.signal_generator(
        patient_generator=uniform_id_generator(ds.get_test_patient_ids(), repeat=False),
        frame_size=params.demo_size,
        samples_per_patient=5,
        target_rate=params.sampling_rate,
    )
    x = next(ds_gen)
    # Run inference
    runner.open()
    logger.debug("Running inference")
    y_pred = np.zeros(x.size, dtype=np.int32)
    for i in tqdm(range(0, x.size, params.frame_size), desc="Inference"):
        if i + params.frame_size > x.size:
            start, stop = x.size - params.frame_size, x.size
        else:
            start, stop = i, i + params.frame_size
        xx = x[start:stop]
        yy = np.zeros(shape=class_shape, dtype=np.int32)
        xx = augment(x=xx, augmentations=params.augmentations, sample_rate=params.sampling_rate)
        xx = preprocess(xx, sample_rate=params.sampling_rate, preprocesses=params.preprocesses)
        xx = xx.reshape(feat_shape)
        runner.set_inputs(xx)
        runner.perform_inference()
        yy = runner.get_outputs()
        y_pred[start:stop] = np.argmax(yy, axis=-1).flatten()
    # END FOR
    runner.close()

    # Report
    logger.debug("Generating report")
    ts = np.arange(0, x.size) / params.sampling_rate

    fig = make_subplots(
        rows=2,
        cols=3,
        specs=[
            [{"colspan": 3, "type": "xy", "secondary_y": True}, None, None],
            [{"type": "xy"}, {"type": "bar"}, {"type": "table"}],
        ],
        subplot_titles=(f"{signal_type} Plot", "IBI Poincare Plot", "HRV Frequency Bands"),
        horizontal_spacing=0.1,
        vertical_spacing=0.2,
    )
    if signal_type == "ECG":
        # Extract R peaks from QRS segments
        pred_bounds = np.concatenate(([0], np.diff(y_pred).nonzero()[0] + 1, [y_pred.size - 1]))
        peaks = []
        for i in range(1, len(pred_bounds)):
            start, stop = pred_bounds[i - 1], pred_bounds[i]
            duration = 1000 * (stop - start) / params.sampling_rate
            if y_pred[start] == params.class_map.get(HKSegment.qrs, -1) and (duration > 20):
                peaks.append(start + np.argmax(np.abs(x[start:stop])))
            # END IF
        # END FOR
        peaks = np.array(peaks)

    elif signal_type == "PPG":
        # peaks = pk.ppg.find_peaks(x, sample_rate=params.sampling_rate)
        pred_bounds = np.concatenate(([0], np.diff(y_pred).nonzero()[0] + 1, [y_pred.size - 1]))
        peaks = []
        for i in range(1, len(pred_bounds)):
            start, stop = pred_bounds[i - 1], pred_bounds[i]
            duration = 1000 * (stop - start) / params.sampling_rate
            if y_pred[start] == params.class_map.get(HKSegment.systolic, -1) and (duration > 100):
                peaks.append(start + np.argmax(np.abs(x[start:stop])))
            # END IF
        # END FOR
        peaks = np.array(peaks)
    else:
        raise ValueError(f"Unknown signal type: {signal_type}")

    band_names = ["VLF", "LF", "HF", "VHF"]
    bands = [(0.0033, 0.04), (0.04, 0.15), (0.15, 0.4), (0.4, 0.5)]

    # Compute R-R intervals
    rri = pk.ecg.compute_rr_intervals(peaks)
    mask = pk.ecg.filter_rr_intervals(rri, sample_rate=params.sampling_rate)
    rri_ms = 1000 * rri / params.sampling_rate
    # Compute metrics
    if (rri.size <= 2) or (mask.sum() / mask.size > 0.80):
        logger.warning("High percentage of RR intervals were filtered out")
        hr_bpm = 0
        hrv_td = pk.hrv.HrvTimeMetrics()
        hrv_fd = pk.hrv.HrvFrequencyMetrics(bands=[pk.hrv.HrvFrequencyBandMetrics() for b in bands])
    else:
        hr_bpm = 60 / (np.nanmean(rri[mask == 0]) / params.sampling_rate)
        hrv_td = pk.hrv.compute_hrv_time(rri[mask == 0], sample_rate=params.sampling_rate)
        hrv_fd = pk.hrv.compute_hrv_frequency(
            peaks[mask == 0], rri[mask == 0], bands=bands, sample_rate=params.sampling_rate
        )
    # END IF

    fig.add_trace(
        go.Scatter(
            x=ts,
            y=x,
            name="SIGNAL",
            mode="lines",
            line=dict(color=primary_color, width=2),
        ),
        row=1,
        col=1,
        secondary_y=False,
    )

    for i, peak in enumerate(peaks):
        color = "red" if mask[i] else "white"
        fig.add_vline(
            x=ts[peak],
            line_width=1,
            line_dash="dash",
            line_color=color,
            annotation={"text": "R-Peak", "textangle": -90, "font_color": color},
            row=1,
            col=1,
            secondary_y=False,
        )
    fig.update_xaxes(title_text="Time (s)", row=1, col=1)
    fig.update_yaxes(title_text=f"{signal_type}", row=1, col=1)

    for i, label in enumerate(classes):
        if label < 0:
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

    fig.add_trace(
        go.Scatter(
            x=rri_ms[:-1],
            y=rri_ms[1:],
            mode="markers",
            marker_size=10,
            showlegend=False,
            customdata=np.arange(1, rri_ms.size),
            hovertemplate="RRn: %{x:.1f} ms<br>RRn+1: %{y:.1f} ms<br>n: %{customdata}",
            marker_color=secondary_color,
        ),
        row=2,
        col=1,
        secondary_y=False,
    )
    if rri_ms.size > 2:
        rr_min, rr_max = np.nanmin(rri_ms) - 20, np.nanmax(rri_ms) + 20
    else:
        rr_min, rr_max = 0, 2000
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
                    [
                        f"{hr_bpm:.0f} BPM",
                        f"{hrv_td.mean_nn:.1f} ms",
                        f"{hrv_td.sd_nn:.1f} ms",
                        f"{hrv_td.rms_sd:.1f}",
                    ],
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
    logger.debug(f"Report saved to {params.job_dir / 'demo.html'}")

    if params.display_report:
        fig.show()
