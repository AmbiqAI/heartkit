import random

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from tqdm import tqdm
import helia_edge as helia

from ...defines import HKTaskParams
from ...backends import BackendFactory
from ...datasets import DatasetFactory, create_augmentation_pipeline
from ...utils import setup_plotting


def demo(params: HKTaskParams):
    """Run denoise demo.

    Args:
        params (HKTaskParams): Task parameters
    """
    logger = helia.utils.setup_logger(__name__, level=params.verbose)

    plot_theme = setup_plotting()

    params.demo_size = params.demo_size or 10 * params.sampling_rate

    # Load backend inference engine
    runner = BackendFactory.get(params.backend)(params=params)

    # Load data
    datasets = [DatasetFactory.get(ds.name)(cacheable=False, **ds.params) for ds in params.datasets]
    ds = random.choice(datasets)

    ds_gen = ds.signal_generator(
        patient_generator=helia.utils.uniform_id_generator(ds.get_test_patient_ids(), repeat=False),
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
        logger.debug(f"Trial {trial + 1}: Cosine Similarity: {cos_sim:.2%} (diff: {cos_sim_diff:.2%})")
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
            line=dict(color=plot_theme.tertiary_color, width=3),
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
            line=dict(color=plot_theme.primary_color, width=3),
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
            line=dict(color=plot_theme.quaternary_color, width=3),
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

    fig.update_layout(
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template=plot_theme.plotly_template,
        height=800,
        plot_bgcolor=plot_theme.bg_color,
        paper_bgcolor=plot_theme.bg_color,
        margin=dict(l=10, r=10, t=80, b=60),
        title="heartKIT: Denoising Demo",
    )

    fig.write_html(params.job_dir / "demo.html", include_plotlyjs="cdn", full_html=False)
    logger.debug(f"Report saved to {params.job_dir / 'demo.html'}")

    if params.display_report:
        fig.show()

    fig, ax = plt.subplots(3, 1, figsize=(10, 6), sharex=True)

    ax[0].plot(ts, y_act, color=plot_theme.tertiary_color, linewidth=3)
    ax[0].set_ylabel("REF")

    ax[1].plot(ts, x, color=plot_theme.primary_color, linewidth=3)
    ax[1].set_ylabel("NOISE")

    ax[2].plot(ts, y_pred, color=plot_theme.quaternary_color, linewidth=3)
    ax[2].set_ylabel("CLEAN")
    ax[2].set_xlabel("Time (s)")

    # Add annotations
    ax[1].annotate(
        f"COS: {cos_sim_orig:.0%}",
        xy=(0.99, 0.05),
        xycoords="axes fraction",
        xytext=(0, 0),
        textcoords="offset points",
        ha="right",
        va="bottom",
        fontsize=14,
        color=plot_theme.fg_color,
        fontweight="bold",
    )
    ax[2].annotate(
        f"COS: {cos_sim:.0%}",
        xy=(0.99, 0.05),
        xycoords="axes fraction",
        xytext=(0, 0),
        textcoords="offset points",
        ha="right",
        va="bottom",
        fontsize=14,
        color=plot_theme.fg_color,
        fontweight="bold",
    )

    fig.tight_layout()
    fig.savefig(params.job_dir / "demo.png")
