import random

import keras
import numpy as np
import numpy.typing as npt
import physiokit as pk
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from tqdm import tqdm
import helia_edge as helia

from ...datasets import IcentiaDataset, PtbxlDataset, DatasetFactory, create_augmentation_pipeline
from ...defines import HKTaskParams
from ...backends import BackendFactory

logger = helia.utils.setup_logger(__name__)


def get_ptbxl_patient_data(
    ds: PtbxlDataset, patient_id: str, frame_size: int, target_rate: int | None = None
) -> tuple[npt.NDArray, npt.NDArray]:
    """Get patient data from PTB-XL dataset.

    Args:
        ds (PtbxlDataset): PTB-XL dataset
        patient_id (str): Patient ID
        frame_size (int): Frame size
        target_rate (int, optional): Target rate. Defaults to None.

    Returns:
        Tuple[np.ndarray, np.ndarray]: ECG data and beat labels
    """
    with ds.patient_data(patient_id=patient_id) as h5:
        data = h5["data"][:]
        blabels = h5[ds.label_key("beat")][:, 0] * 5  # Stored in 100Hz
    # END WITH
    input_size = int(np.ceil((ds.sampling_rate / target_rate) * frame_size))
    lead = random.choice(ds.leads)
    start = np.random.randint(0, data.shape[1] - input_size)
    x = data[lead, start : start + input_size].squeeze()
    x = np.nan_to_num(x).astype(np.float32)
    y = blabels[(blabels >= start) & (blabels < start + input_size)] - start
    if ds.sampling_rate != target_rate:
        ratio = target_rate / ds.sampling_rate
        x = pk.signal.resample_signal(x, ds.sampling_rate, target_rate, axis=0)
        x = x[:frame_size]  # truncate to frame size
        y = (y * ratio).astype(np.int32)
    # END IF
    return x, y


def get_icentia11k_patient_data(
    ds: IcentiaDataset, patient_id: str, frame_size: int, target_rate: int | None = None
) -> tuple[npt.NDArray, npt.NDArray]:
    """Get patient data from Icentia11k dataset.

    Args:
        ds (IcentiaDataset): Icentia11k dataset
        patient_id (str): Patient ID
        frame_size (int): Frame size
        target_rate (int, optional): Target rate. Defaults to None.

    Returns:
        Tuple[np.ndarray, np.ndarray]: ECG data and beat labels
    """

    if target_rate is None:
        target_rate = ds.sampling_rate

    input_size = int(np.ceil((ds.sampling_rate / target_rate) * frame_size))
    label_key = ds.label_key("beat")

    with ds.patient_data(patient_id) as segments:
        segment = segments[np.random.choice(list(segments.keys()))]
        segment_size = segment["data"].shape[0]
        frame_start = np.random.randint(segment_size - input_size)
        frame_end = frame_start + input_size
        x = segment["data"][frame_start:frame_end].squeeze()
        x = np.nan_to_num(x).astype(np.float32)

        labels = segment[label_key]
        labels = labels[(labels[:, 0] >= frame_start) & (labels[:, 0] < frame_end) & (labels[:, 1] != 0)]
        y = labels[:, 0] - frame_start
        if ds.sampling_rate != target_rate:
            ratio = target_rate / ds.sampling_rate
            x = pk.signal.resample_signal(x, ds.sampling_rate, target_rate, axis=0)
            x = x[:frame_size]  # truncate to frame size
            y = (y * ratio).astype(np.int32)
        # END IF

    # END WITH
    return x, y


def demo(params: HKTaskParams):
    """Run demo on model.

    Args:
        params (HKTaskParams): Demo parameters
    """

    bg_color = "rgba(38,42,50,1.0)"
    primary_color = "#11acd5"
    secondary_color = "#ce6cff"
    tertiary_color = "rgb(234,52,36)"
    quaternary_color = "rgb(92,201,154)"
    plotly_template = "plotly_dark"
    marker_colors = [primary_color, secondary_color, tertiary_color, quaternary_color]

    params.demo_size = params.demo_size or 20 * params.sampling_rate

    # Load backend inference engine
    runner = BackendFactory.get(params.backend)(params=params)

    # Load data
    class_names = params.class_names or [f"Class {i}" for i in range(params.num_classes)]

    feat_shape = (params.frame_size, 1)

    dsets = [DatasetFactory.get(ds.name)(**ds.params) for ds in params.datasets]

    ds = random.choice(dsets)
    if ds.name == "ptbxl":
        pt_id = random.choice(ds.get_test_patient_ids())
        x, peaks = get_ptbxl_patient_data(
            ds,
            patient_id=pt_id,
            frame_size=params.demo_size,
            target_rate=params.sampling_rate,
        )
    elif ds.name == "icentia11k":
        pt_id = random.choice(ds.get_test_patient_ids())
        x, peaks = get_icentia11k_patient_data(
            ds,
            patient_id=pt_id,
            frame_size=params.demo_size,
            target_rate=params.sampling_rate,
        )
    else:
        # Need to manually locate peaks, compute
        ds_gen = ds.signal_generator(
            patient_generator=helia.utils.uniform_id_generator(ds.get_test_patient_ids(), repeat=False),
            frame_size=params.demo_size,
            samples_per_patient=5,
            target_rate=params.sampling_rate,
        )
        x = next(ds_gen)
        xf = pk.ecg.clean(x, sample_rate=params.sampling_rate)
        xf = pk.signal.normalize_signal(xf, eps=0.1, axis=None)
        peaks = pk.ecg.find_peaks(xf, sample_rate=params.sampling_rate)
    # END IF

    rri = pk.ecg.compute_rr_intervals(peaks)

    augmenter = create_augmentation_pipeline(
        params.augmentations + params.preprocesses,
        sampling_rate=params.sampling_rate,
    )

    # Run inference
    runner.open()
    logger.debug("Running inference")
    y_prob = np.zeros_like(peaks, dtype=np.float32)
    y_pred = np.zeros_like(peaks, dtype=np.int32)
    for i in tqdm(range(0, len(peaks)), desc="Inference"):
        start = peaks[i] - int(params.frame_size * 0.5)
        stop = start + params.frame_size
        if start < 0 or stop > x.size:
            y_pred[i] = -1
            y_prob[i] = 0.0
            continue
        xx = x[start:stop]
        xx = xx.reshape(feat_shape)
        xx = augmenter(xx)
        runner.set_inputs(xx)
        runner.perform_inference()
        yy = runner.get_outputs()
        yy = keras.ops.softmax(yy).numpy()
        y_pred[i] = np.argmax(yy, axis=-1)
        y_prob[i] = yy[y_pred[i]]
        if y_prob[i] < params.threshold:
            y_pred[i] = 0
        # END IF
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
            [{"type": "xy"}, {"type": "bar", "colspan": 2}, None],
        ],
        subplot_titles=("ECG Plot", "IBI Poincare Plot", "Beat Counts"),
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
        if y_pred[i] < 0:
            continue
        color = "white" if y_pred[i] == 0 else secondary_color
        label = f"{class_names[y_pred[i]]} ({y_prob[i]:0.0%})"
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
    fig.update_xaxes(
        title_text="Time (s)",
        row=1,
        col=1,
        autorangeoptions=dict(
            minallowed=0.5 * params.frame_size / params.sampling_rate,
            maxallowed=ts[-1] - 0.5 * params.frame_size / params.sampling_rate,
        ),
    )
    fig.update_yaxes(title_text="ECG", row=1, col=1)

    # Plot normal beats overlayed

    # Plot poincare plot of RR intervals with color for PAC/PVC
    rri_ms = 1000 * rri / params.sampling_rate
    fig.add_trace(
        go.Scatter(
            x=[rri_ms[i] if y_pred[i] >= 0 else None for i in range(len(rri) - 1)],
            y=[rri_ms[i + 1] if y_pred[i + 1] >= 0 else None for i in range(len(rri) - 1)],
            # y=rri_ms[1:],
            mode="markers",
            marker_size=10,
            showlegend=False,
            text=[class_names[y_pred[i]] if y_pred[i] >= 0 else "" for i in range(1, len(rri) - 1)],
            marker_color=[secondary_color if y_pred[i] > 0 else primary_color for i in range(1, len(rri) - 1)],
        ),
        row=2,
        col=1,
        secondary_y=False,
    )
    if rri_ms.size:
        rr_min, rr_max = np.nanmin(rri_ms) - 20, np.nanmax(rri_ms) + 20
    else:
        rr_min, rr_max = 0, 1000
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

    # Plot beat frequency counts

    fig.add_trace(
        go.Bar(
            x=np.bincount(y_pred[y_pred >= 0], minlength=params.num_classes),
            y=class_names,
            marker_color=[marker_colors[i % 4] for i in range(params.num_classes)],
            orientation="h",
            showlegend=False,
        ),
        row=2,
        col=2,
    )
    fig.update_xaxes(title_text="Number of Beats", row=2, col=2)

    fig.update_layout(
        template=plotly_template,
        height=800,
        plot_bgcolor=bg_color,
        paper_bgcolor=bg_color,
        margin=dict(l=10, r=10, t=80, b=80),
        legend=dict(groupclick="toggleitem"),
        title="heartKIT: Beat Demo",
    )

    fig.write_html(params.job_dir / "demo.html", include_plotlyjs="cdn", full_html=True)
    logger.debug(f"Report saved to {params.job_dir / 'demo.html'}")

    if params.display_report:
        fig.show()
