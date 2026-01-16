import random

import numpy as np
import numpy.typing as npt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.manifold import TSNE
from tqdm import tqdm
import helia_edge as helia

from ...defines import HKTaskParams
from ...backends import BackendFactory
from ...datasets import DatasetFactory
from .dataloaders import FoundationTaskFactory
from ...utils import setup_plotting


def demo(params: HKTaskParams):
    """Run demo for model

    Args:
        params (HKTaskParams): Task parameters
    """
    logger = helia.utils.setup_logger(__name__, level=params.verbose)
    plot_theme = setup_plotting()

    feat_shape = (params.frame_size, 1)
    num_pts = min(params.batch_size, 256)

    # Load backend inference engine
    runner = BackendFactory.get(params.backend)(params=params)

    # load datasets and randomly select one
    dataset = random.choice(params.datasets)
    ds = DatasetFactory.get(dataset.name)(**dataset.params)
    dataloader = FoundationTaskFactory.get(dataset.name)(
        ds=ds,
        frame_size=params.frame_size,
        sampling_rate=params.sampling_rate,
    )
    patients: npt.NDArray = ds.get_test_patient_ids()
    patients = np.random.choice(patients, size=num_pts, replace=False)

    x1 = np.zeros((num_pts, *feat_shape), dtype=np.float32)
    x2 = np.zeros((num_pts, *feat_shape), dtype=np.float32)

    # For each patient, generate TGT_LEN samples
    i = 0
    for patient in patients:
        for xx1, xx2 in dataloader.patient_data_generator(patient, 1):
            x1[i] = xx1
            x2[i] = xx2
            i += 1
        # END FOR
    # END FOR

    # Run inference
    runner.open()
    logger.debug("Running inference")
    y1 = np.zeros((num_pts, params.num_classes), dtype=np.float32)
    y2 = np.zeros((num_pts, params.num_classes), dtype=np.float32)
    for i in tqdm(range(0, num_pts), desc="Inference"):
        runner.set_inputs(x1[i])
        runner.perform_inference()
        y1[i] = runner.get_outputs()
        runner.set_inputs(x2[i])
        runner.perform_inference()
        y2[i] = runner.get_outputs()
    # END FOR
    runner.close()

    def cossim(x, y):
        return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))

    # END DEF

    pts_cosim = np.zeros((num_pts, num_pts), dtype=np.float32)
    # Compute cosine similarity between pairs of patients
    for i in range(num_pts):
        for j in range(num_pts):
            pts_cosim[i, j] = cossim(y1[i], y1[j])
        # END FOR
    # END FOR

    # Get average cosine similarity down the diagonal
    mu_tgt_cos = np.mean(np.diag(pts_cosim))
    mu_other = np.mean(pts_cosim[np.eye(num_pts) == 0])

    logger.info(f"Average cosine similarity (target): {mu_tgt_cos:0.2%}")
    logger.info(f"Average cosine similarity (other): {mu_other:0.2%}")

    # Compute t-SNE
    tsne = TSNE(n_components=2, random_state=0, n_iter=5000, perplexity=75)
    x_tsne = tsne.fit_transform(np.concatenate([y1, y2]))

    fig = make_subplots(
        rows=1,
        cols=2,
        specs=[
            [{"type": "xy"}, {"type": "heatmap"}],
        ],
        subplot_titles=("t-SNE", "Cosine Similarity"),
    )

    pts = np.concatenate([patients, patients]).tolist()
    fig.add_trace(
        go.Scatter(
            x=x_tsne[:, 0],
            y=x_tsne[:, 1],
            mode="markers",
            customdata=pts,
            hovertemplate="Patient %{customdata}<extra></extra>",
            marker_color=[px.colors.qualitative.Dark24[i % 24] for i in pts],
            marker_size=8,
            name="t-SNE",
        ),
        row=1,
        col=1,
    )

    # Add heatmap of cosine similarity
    fig.add_trace(
        go.Heatmap(
            z=pts_cosim,
            x=list(range(num_pts)),
            y=list(range(num_pts)),
            colorscale="Plotly3",
            showscale=True,
            zmin=0,
            zmax=1,
        ),
        row=1,
        col=2,
    )

    fig.update_layout(
        template=plot_theme.plotly_template,
        height=600,
        plot_bgcolor=plot_theme.bg_color,
        paper_bgcolor=plot_theme.bg_color,
        margin=dict(l=10, r=10, t=80, b=80),
        legend=dict(groupclick="toggleitem"),
        title="heartKIT: Foundation Demo",
    )
    fig.write_html(params.job_dir / "demo.html", include_plotlyjs="cdn", full_html=True)
    logger.debug(f"Report saved to {params.job_dir / 'demo.html'}")

    if params.display_report:
        fig.show()
