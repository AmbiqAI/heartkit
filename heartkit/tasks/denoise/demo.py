import random

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from tqdm import tqdm
import neuralspot_edge as nse

from ...defines import HKTaskParams
from ...rpc import BackendFactory
from ...datasets import DatasetFactory, create_augmentation_pipeline


def demo(params: HKTaskParams):
    """Run segmentation demo.

    Args:
        params (HKTaskParams): Demo parameters
    """
    logger = nse.utils.setup_logger(__name__, level=params.verbose)

    bg_color = "rgba(38,42,50,1.0)"
    primary_color = "#11acd5"
    # secondary_color = "#ce6cff"
    tertiary_color = "rgb(234,52,36)"
    quaternary_color = "rgb(92,201,154)"
    plotly_template = "plotly_dark"

    params.demo_size = params.demo_size or 10 * params.sampling_rate

    # Load backend inference engine
    runner = BackendFactory.get(params.backend)(params=params)

    # Load data
    datasets = [DatasetFactory.get(ds.name)(cacheable=False, **ds.params) for ds in params.datasets]
    ds = random.choice(datasets)

    ds_gen = ds.signal_generator(
        patient_generator=nse.utils.uniform_id_generator(ds.get_test_patient_ids(), repeat=False),
        frame_size=params.demo_size,
        samples_per_patient=5,
        target_rate=params.sampling_rate,
    )
    x = next(ds_gen)
    x = np.nan_to_num(x, neginf=0, posinf=0).astype(np.float32)
    x = np.reshape(x, (-1, 1))
    y_act = x.copy()

    preprocessor = create_augmentation_pipeline(
        params.preprocesses,
        sampling_rate=params.sampling_rate,
    )
    augmenter = create_augmentation_pipeline(
        params.augmentations,
        sampling_rate=params.sampling_rate,
    )

    x = preprocessor(augmenter(x)).numpy()
    y_act = preprocessor(y_act).numpy()

    x = x.flatten()
    y_act = y_act.flatten()

    # Run inference
    runner.open()
    logger.debug("Running inference")
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
        cos_sim = np.dot(y_act, y_pred) / (np.linalg.norm(y_act) * np.linalg.norm(y_pred))
        cos_sim_diff = cos_sim - prev_cos_sim
        prev_cos_sim = cos_sim
        logger.debug(f"Trial {trial+1}: Cosine Similarity: {cos_sim:.2%} (diff: {cos_sim_diff:.2%})")
        if cos_sim_diff < 1e-3:
            break
    # END FOR

    # END FOR
    runner.close()
    # Report
    logger.debug("Generating report")
    ts = np.arange(0, x.size) / params.sampling_rate

    # Compute cosine similarity
    cos_sim_orig = np.dot(y_act, x) / (np.linalg.norm(y_act) * np.linalg.norm(x))
    cos_sim = np.dot(y_act, y_pred) / (np.linalg.norm(y_act) * np.linalg.norm(y_pred))
    logger.debug(f"Before Cosine Similarity: {cos_sim_orig:.2%}")
    logger.debug(f"After Cosine Similarity: {cos_sim:.2%}")

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
            y=y_act,
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
            name="NOISE",
            mode="lines",
            line=dict(color=primary_color, width=3),
        ),
        row=2,
        col=1,
    )
    fig.update_yaxes(title_text="NOISE", row=2, col=1)

    fig.add_trace(
        go.Scatter(
            x=ts,
            y=y_pred,
            name="CLEAN",
            mode="lines",
            line=dict(color=quaternary_color, width=3),
        ),
        row=3,
        col=1,
    )
    fig.update_yaxes(title_text="CLEAN", row=3, col=1)

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
        title="HeartKit: Denoising Demo",
    )

    fig.write_html(params.job_dir / "demo.html", include_plotlyjs="cdn", full_html=False)
    logger.debug(f"Report saved to {params.job_dir / 'demo.html'}")

    if params.display_report:
        fig.show()
