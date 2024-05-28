import random

import numpy as np
import plotly.graph_objects as go
import tensorflow as tf
from plotly.subplots import make_subplots
from tqdm import tqdm

from ...datasets.utils import uniform_id_generator
from ...defines import HKDemoParams
from ...rpc import BackendFactory
from ...utils import setup_logger
from ..utils import load_datasets
from .datasets import get_data_generator, prepare

logger = setup_logger(__name__)


def demo(params: HKDemoParams):
    """Run task demo.

    Args:
        params (HKDemoParams): Demo parameters
    """
    bg_color = "rgba(38,42,50,1.0)"
    primary_color = "#11acd5"
    # secondary_color = "#ce6cff"
    tertiary_color = "rgb(234,52,36)"
    quaternary_color = "rgb(92,201,154)"
    plotly_template = "plotly_dark"

    params.demo_size = params.demo_size or 10 * params.sampling_rate

    # Load backend inference engine
    runner = BackendFactory.create(params.backend, params=params)

    feat_shape = (params.demo_size, 1)
    class_shape = (params.demo_size, 1)

    ds_spec = (
        tf.TensorSpec(shape=feat_shape, dtype=tf.float32),
        tf.TensorSpec(shape=class_shape, dtype=tf.float32),
    )

    # Load data
    dsets = load_datasets(datasets=params.datasets)
    ds = random.choice(dsets)

    ds_gen = get_data_generator(
        ds, frame_size=params.demo_size, samples_per_patient=5, target_rate=params.sampling_rate
    )

    ds_gen = ds_gen(patient_generator=uniform_id_generator(ds.get_test_patient_ids(), repeat=False))

    x, y = next(ds_gen)

    x, y = prepare(
        (x, y),
        sample_rate=params.sampling_rate,
        preprocesses=params.preprocesses,
        augmentations=params.augmentations,
        spec=ds_spec,
        num_classes=params.num_classes,
    )
    x = x.flatten()
    y = y.flatten()

    # Run inference
    runner.open()
    logger.info("Running inference")
    y_pred = np.zeros(x.size, dtype=np.float32)

    cos_sim_diff = 0
    prev_cos_sim = 0

    x_input = x.copy()
    for trial in range(8):
        for i in tqdm(range(0, x.size, params.frame_size), desc="Inference"):
            if i + params.frame_size > x.size:
                start, stop = x.size - params.frame_size, x.size
            else:
                start, stop = i, i + params.frame_size
            xx = x_input[start:stop]
            runner.set_inputs(xx)
            runner.perform_inference()
            yy = runner.get_outputs()
            y_pred[start:stop] = yy.flatten()
        # END FOR
        x_input = y_pred.copy()
        cos_sim = np.dot(y, y_pred) / (np.linalg.norm(y) * np.linalg.norm(y_pred))
        cos_sim_diff = cos_sim - prev_cos_sim
        prev_cos_sim = cos_sim
        logger.info(f"Trial {trial+1}: Cosine Similarity: {cos_sim:.2%} (diff: {cos_sim_diff:.2%})")
        if cos_sim_diff < 1e-3:
            break
    # END FOR

    # END FOR
    runner.close()
    # Report
    logger.info("Generating report")
    ts = np.arange(0, x.size) / params.sampling_rate

    # Compute cosine similarity
    cos_sim_orig = np.dot(y, x) / (np.linalg.norm(y) * np.linalg.norm(x))
    cos_sim = np.dot(y, y_pred) / (np.linalg.norm(y) * np.linalg.norm(y_pred))
    logger.info(f"Before Cosine Similarity: {cos_sim_orig:.2%}")
    logger.info(f"After Cosine Similarity: {cos_sim:.2%}")

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        horizontal_spacing=0.1,
        vertical_spacing=0.05,
    )

    fig.add_trace(
        go.Scatter(
            x=ts,
            y=y,
            name="REF",
            mode="lines",
            line=dict(color=tertiary_color, width=3),
        ),
        row=1,
        col=1,
    )
    fig.update_yaxes(title_text="REF", row=1, col=1)

    fig.add_trace(
        go.Scatter(
            x=ts,
            y=x,
            name="INPUT",
            mode="lines",
            line=dict(color=primary_color, width=3),
        ),
        row=2,
        col=1,
    )
    fig.update_yaxes(title_text="INPUT", row=2, col=1)

    fig.add_trace(
        go.Scatter(
            x=ts,
            y=y_pred,
            name="OUTPUT",
            mode="lines",
            line=dict(color=quaternary_color, width=3),
        ),
        row=3,
        col=1,
    )
    fig.update_yaxes(title_text="OUTPUT", row=3, col=1)

    fig.add_annotation(
        x=1,
        y=0.3,
        text=f"CoSim: {cos_sim_orig:.2%}",
        showarrow=False,
        xref="paper",
        yref="paper",
        align="right",
        font=dict(
            family="Menlo",
            size=24,
        ),
    )

    fig.add_annotation(
        x=1,
        y=-0.08,
        text=f"CoSim: {cos_sim:.2%}",
        showarrow=False,
        xref="paper",
        yref="paper",
        align="right",
        font=dict(
            family="Menlo",
            size=24,
        ),
    )

    fig.update_xaxes(title_text="Time (s)", row=3, col=1)
    # fig.update_yaxes(title_text="SIGNAL", row=1, col=1)

    fig.update_layout(
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template=plotly_template,
        height=800,
        plot_bgcolor=bg_color,
        paper_bgcolor=bg_color,
        margin=dict(l=10, r=10, t=80, b=60),
        title="HeartKit: Translate Demo",
    )

    fig.write_html(params.job_dir / "demo.html", include_plotlyjs="cdn", full_html=False)
    logger.info(f"Report saved to {params.job_dir / 'demo.html'}")

    if params.display_report:
        fig.show()
