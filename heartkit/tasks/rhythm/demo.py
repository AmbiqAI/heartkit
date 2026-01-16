import datetime
import random

import keras
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from tqdm import tqdm
import helia_edge as helia

from ...defines import HKTaskParams
from ...backends import BackendFactory
from ...datasets import DatasetFactory, create_augmentation_pipeline
from ...utils import setup_plotting


def demo(params: HKTaskParams):
    """Run demo for model

    Args:
        params (HKTaskParams): Task parameters
    """
    logger = helia.utils.setup_logger(__name__, level=params.verbose)

    plot_theme = setup_plotting()

    params.demo_size = params.demo_size or 2 * params.frame_size

    # Load backend inference engine
    runner = BackendFactory.get(params.backend)(params=params)

    # Load data
    # classes = sorted(list(set(params.class_map.values())))
    class_names = params.class_names or [f"Class {i}" for i in range(params.num_classes)]

    feat_shape = (params.frame_size, 1)

    datasets = [DatasetFactory.get(ds.name)(**ds.params) for ds in params.datasets]
    ds = random.choice(datasets)

    ds_gen = ds.signal_generator(
        patient_generator=helia.utils.uniform_id_generator(ds.get_test_patient_ids(), repeat=False),
        frame_size=params.demo_size,
        samples_per_patient=5,
        target_rate=params.sampling_rate,
    )
    x = next(ds_gen)

    augmenter = create_augmentation_pipeline(
        params.preprocesses + params.augmentations, sampling_rate=params.sampling_rate
    )

    # Run inference
    runner.open()
    logger.debug("Running inference")
    y_preds: list[tuple[int, int, int, float]] = []  # (start, stop, class, prob)
    for i in tqdm(range(0, x.shape[0], params.frame_size), desc="Inference"):
        if i + params.frame_size > x.shape[0]:
            start, stop = x.shape[0] - params.frame_size, x.shape[0]
        else:
            start, stop = i, i + params.frame_size
        xx = x[start:stop]
        xx = xx.reshape(feat_shape)
        xx = augmenter(xx, training=False)
        runner.set_inputs(xx)
        runner.perform_inference()
        yy = runner.get_outputs()
        y_prob = np.exp(yy) / np.sum(np.exp(yy), axis=-1, keepdims=True)
        y_pred = np.argmax(y_prob, axis=-1)
        y_prob = y_prob[y_pred]
        if y_prob < params.threshold:
            y_pred = -1
        y_preds.append((start, stop - 1, y_pred, y_prob))
    # END FOR
    runner.close()

    # Report
    logger.debug("Generating report")
    tod = datetime.datetime(2025, 5, 24, random.randint(12, 23), 00)
    ts = np.array([tod + datetime.timedelta(seconds=i / params.sampling_rate) for i in range(x.shape[0])])

    fig = make_subplots(
        rows=1,
        cols=1,
        specs=[
            [{"colspan": 1, "type": "xy", "secondary_y": True}],
        ],
        subplot_titles=(None, None),
        horizontal_spacing=0.05,
        vertical_spacing=0.1,
    )

    fig.add_trace(
        go.Scatter(
            x=ts,
            y=x,
            name="ECG",
            mode="lines",
            line=dict(color=plot_theme.primary_color, width=2),
            showlegend=False,
        ),
        row=1,
        col=1,
        secondary_y=False,
    )

    for i, (start, stop, pred, prob) in enumerate(y_preds):
        if pred < 0:
            label = f"Inconclusive ({prob:0.0%})"
        else:
            label = f"{class_names[pred]} ({prob:0.0%})"
        color = plot_theme.colors[pred % len(plot_theme.colors)]
        if i > 0 and y_preds[i - 1][1] >= start:
            start = y_preds[i - 1][1] + 1
        fig.add_vrect(
            x0=ts[start],
            x1=ts[stop],
            annotation_text=label,
            fillcolor=color,
            opacity=0.25,
            line_width=2,
            line_color=color,
            row=1,
            col=1,
            secondary_y=False,
        )

    fig.update_layout(
        template=plot_theme.plotly_template,
        height=600,
        plot_bgcolor=plot_theme.bg_color,
        paper_bgcolor=plot_theme.bg_color,
        margin=dict(l=10, r=10, t=80, b=80),
        legend=dict(groupclick="toggleitem"),
        title="heartKIT: Rhythm Demo",
    )
    fig.write_html(params.job_dir / "demo.html", include_plotlyjs="cdn", full_html=False)
    logger.debug(f"Report saved to {params.job_dir / 'demo.html'}")

    # Matplotlib version

    # Make ts in seconds
    ts = np.arange(x.shape[0]) / params.sampling_rate
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(ts, x, color=plot_theme.primary_color, linewidth=2)
    for i, (start, stop, pred, prob) in enumerate(y_preds):
        if pred < 0:
            label = f"Inconclusive ({prob:0.0%})"
        else:
            label = f"{class_names[pred]} ({prob:0.0%})"
        color = plot_theme.colors[pred % len(plot_theme.colors)]
        ax.axvspan(ts[start], ts[stop], color=color, alpha=0.25, label=label)
    ax.set_title("heartKIT: Rhythm Demo")
    ax.set_xlabel("Time (sec)")
    ax.set_ylabel("ECG")
    ax.legend(loc="upper right", bbox_to_anchor=(1, 1), frameon=False)
    fig.tight_layout()
    fig.savefig(params.job_dir / "demo.png")

    if params.display_report:
        fig.show()

    # cleanup
    keras.utils.clear_session()
    for ds in datasets:
        ds.close()
