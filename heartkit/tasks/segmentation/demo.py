import random

import numpy as np
import physiokit as pk
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from tqdm import tqdm
import helia_edge as helia

from ...defines import HKTaskParams
from ...backends import BackendFactory
from ...datasets import DatasetFactory, create_augmentation_pipeline
from .defines import HKSegment
from ...utils import setup_plotting


def demo(params: HKTaskParams):
    """Run segmentation demo.

    Args:
        params (HKTaskParams): Demo parameters
    """
    logger = helia.utils.setup_logger(__name__, level=params.verbose)
    plot_theme = setup_plotting()

    signal_type = getattr(params, "signal_type", "ECG").upper()  # ECG or PPG
    params.demo_size = params.demo_size or params.frame_size

    # Load backend inference engine
    runner = BackendFactory.get(params.backend)(params)

    classes = sorted(set(params.class_map.values()))
    class_names = params.class_names or [f"Class {i}" for i in range(params.num_classes)]

    feat_shape = (params.frame_size, 1)
    class_shape = (params.frame_size, params.num_classes)

    datasets = [DatasetFactory.get(ds.name)(cacheable=False, **ds.params) for ds in params.datasets]
    ds = random.choice(datasets)

    ds_gen = ds.signal_generator(
        patient_generator=helia.utils.uniform_id_generator(ds.get_test_patient_ids(), repeat=False),
        frame_size=params.demo_size,
        samples_per_patient=5,
        target_rate=params.sampling_rate,
    )
    x = next(ds_gen)

    augmenter = create_augmentation_pipeline(
        augmentations=params.augmentations + params.preprocesses,
        sampling_rate=params.sampling_rate,
    )

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
        xx = xx.reshape(feat_shape)
        xx = augmenter(xx, training=True)
        runner.set_inputs(xx)
        runner.perform_inference()
        x[start:stop] = xx.numpy().squeeze()
        yy = runner.get_outputs()
        y_pred[start:stop] = np.argmax(yy, axis=-1).flatten()
    # END FOR
    runner.close()

    # Report
    logger.debug("Generating report")
    ts = np.arange(0, x.size) / params.sampling_rate

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
                peaks.append(start + np.argmax(x[start:stop]))
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

    fig.add_trace(
        go.Scatter(
            x=ts,
            y=x,
            name="SIGNAL",
            mode="lines",
            line=dict(color=plot_theme.primary_color, width=2),
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
            marker_color=[
                plot_theme.primary_color,
                plot_theme.secondary_color,
                plot_theme.tertiary_color,
                plot_theme.quaternary_color,
            ],
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
            marker_color=plot_theme.secondary_color,
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
                fill_color=plot_theme.primary_color,
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
                fill_color=plot_theme.bg_color,
                align=["left"],
            ),
        ),
        row=2,
        col=3,
    )

    fig.update_layout(
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template=plot_theme.plotly_template,
        height=800,
        plot_bgcolor=plot_theme.bg_color,
        paper_bgcolor=plot_theme.bg_color,
        margin=dict(l=10, r=10, t=80, b=80),
        title="heartKIT: Segmentation Demo",
    )

    fig.write_html(params.job_dir / "demo.html", include_plotlyjs="cdn", full_html=False)
    logger.debug(f"Report saved to {params.job_dir / 'demo.html'}")

    if params.display_report:
        fig.show()

    # Reproduce above in matplotlib
    # First row is full width
    # Bottom row has 3 columns

    fig = plt.figure(layout="constrained", figsize=(10, 6))
    gs = GridSpec(2, 3, figure=fig)
    ax1 = fig.add_subplot(gs[0, :])
    ax21 = fig.add_subplot(gs[1, 0])
    ax22 = fig.add_subplot(gs[1, 1])
    ax23 = fig.add_subplot(gs[1, 2])

    ax1.plot(ts, x, color=plot_theme.primary_color)

    for i, label in enumerate(classes):
        if label < 0:
            continue
        color = plot_theme.colors[i % len(plot_theme.colors)]
        ax1.plot(ts, np.where(y_pred == label, x, np.nan), label=class_names[i], color=color)
    # END FOR
    # Add R peaks
    for i, peak in enumerate(peaks):
        color = "red" if mask[i] else "white"
        ax1.axvline(x=ts[peak], color=color, linestyle="--", label="R-Peak" if i == 0 else None)
    ax1.set_title(f"{signal_type} Plot")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel(f"{signal_type}")
    # ax1.legend(loc="upper right", ncols=len(class_names))

    # IBI Poincare Plot
    ax21.scatter(rri_ms[:-1], rri_ms[1:], color=plot_theme.secondary_color)
    # Add 45-degree line
    rr_min, rr_max = np.nanmin(rri_ms) - 20, np.nanmax(rri_ms) + 20
    ax21.plot([rr_min, rr_max], [rr_min, rr_max], color="white", linestyle="--")
    ax21.set_title("IBI Poincare Plot")
    ax21.set_xlabel("RRn (ms)")
    ax21.set_ylabel("RRn+1 (ms)")

    # HRV Frequency Bands
    ax22.barh(band_names, np.array([b.total_power for b in hrv_fd.bands]) / hrv_fd.total_power)
    ax22.set_title("HRV Frequency Bands")
    ax22.set_xlabel("Normalized Power")

    # HRV Metrics
    ax23.axis("off")
    table_data = [
        ["Heart Rate", f"{hr_bpm:.0f} BPM"],
        ["NN Mean", f"{hrv_td.mean_nn:.1f} ms"],
        ["NN St. Dev", f"{hrv_td.sd_nn:.1f} ms"],
        ["SD RMS", f"{hrv_td.rms_sd:.1f}"],
    ]
    table = ax23.table(cellText=table_data, cellLoc="left", loc="center", cellColours=[[plot_theme.bg_color] * 2] * 4)
    table.scale(1.5, 3)
    table.auto_set_font_size(True)
    ax23.set_title("HRV Metrics")
    for key, cell in table.get_celld().items():
        cell.set_edgecolor(plot_theme.fg_color)

    fig.tight_layout()
    fig.savefig(params.job_dir / "demo.png")
