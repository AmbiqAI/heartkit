import random

import numpy as np
import numpy.typing as npt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.manifold import TSNE
from tqdm import tqdm
import neuralspot_edge as nse

from ...defines import HKTaskParams
from ...rpc import BackendFactory
from ...datasets import DatasetFactory


def demo(params: HKTaskParams):
    """Run demo for model

    Args:
        params (HKTaskParams): Demo parameters
    """

    logger = nse.utils.setup_logger(__name__, level=params.verbose)

    bg_color = "rgba(38,42,50,1.0)"
    # primary_color = "#11acd5"
    # secondary_color = "#ce6cff"
    plotly_template = "plotly_dark"

    feat_shape = (params.frame_size, 1)
    NUM_PTS = 50
    TGT_LEN = 20

    # Load backend inference engine
    runner = BackendFactory.get(params.backend)(params=params)

    # load datasets and randomly select one
    datasets = [DatasetFactory.get(ds.name)(**ds.params) for ds in params.datasets]
    ds = random.choice(datasets)

    patients: npt.NDArray = ds.get_test_patient_ids()
    patients = np.random.choice(patients, size=NUM_PTS, replace=False)

    x = []
    y = []
    # For each patient, generate TGT_LEN samples
    for i, patient in enumerate(patients):
        ds_gen = ds.signal_generator(
            patient_generator=nse.utils.uniform_id_generator([patient], repeat=False),
            frame_size=params.frame_size,
            samples_per_patient=TGT_LEN,
            target_rate=params.sampling_rate,
        )
        for _ in range(TGT_LEN):
            x.append(next(ds_gen))
            y.append(i)
        # END FOR
    # END FOR

    # Run inference
    runner.open()
    logger.debug("Running inference")
    x_p = []
    for i in tqdm(range(0, len(x)), desc="Inference"):
        # x[i] = preprocess(x[i], sample_rate=params.sampling_rate, preprocesses=params.preprocesses)
        xx = x[i].copy()
        xx = xx.reshape(feat_shape)
        runner.set_inputs(xx)
        runner.perform_inference()
        x_p.append(runner.get_outputs())
    # END FOR
    runner.close()

    def cossim(x, y):
        return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))

    # END DEF

    pts_cosim = np.zeros((NUM_PTS, NUM_PTS), dtype=np.float32)
    # Compute cosine similarity between pairs of patients
    for i, patient in enumerate(patients):
        start_i = i * TGT_LEN
        tgt_sample = x_p[start_i]
        for j, _ in enumerate(patients):
            start_j = j * TGT_LEN
            stop_j = start_j + TGT_LEN
            # Average cosine similarity between start_i and start_j
            avg_cosim = np.mean([cossim(tgt_sample, x_p[k]) for k in range(start_j, stop_j)])
            pts_cosim[i, j] = avg_cosim
        # END FOR
    # END FOR

    # # Get average cosine similarity for target and non-target samples
    # tgt_avg_cosim = np.mean(x_cosim[1:TGT_LEN])
    # rem_avg_cosim = np.mean(x_cosim[TGT_LEN:])
    # tgt_avg_p_cosim = np.mean(x_p_cosim[1:TGT_LEN])
    # rem_avg_p_cosim = np.mean(x_p_cosim[TGT_LEN:])

    # print(f"TARGET AVG PRE-COSSIM: {tgt_avg_cosim:0.2%}")
    # print(f" OTHER AVG PRE-COSSIM: {rem_avg_cosim:0.2%}")
    # print(f"TARGET AVG POST-COSSIM: {tgt_avg_p_cosim:0.2%}")
    # print(f" OTHER AVG POST-COSSIM: {rem_avg_p_cosim:0.2%}")

    # Compute t-SNE
    tsne = TSNE(n_components=2, random_state=0, n_iter=5000, perplexity=75)
    x_tsne = tsne.fit_transform(np.array(x_p).reshape((len(x_p), -1)))

    fig = make_subplots(
        rows=1,
        cols=2,
        specs=[
            [{"type": "xy"}, {"type": "heatmap"}],
        ],
        subplot_titles=("t-SNE", "Cosine Similarity"),
    )

    fig.add_trace(
        go.Scatter(
            x=x_tsne[:, 0],
            y=x_tsne[:, 1],
            mode="markers",
            customdata=[patients[i] for i in y],
            hovertemplate="Patient %{customdata}<extra></extra>",
            marker_color=[px.colors.qualitative.Dark24[i % 24] for i in y],
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
            x=list(range(NUM_PTS)),
            y=list(range(NUM_PTS)),
            colorscale="Plotly3",
            showscale=True,
            zmin=0,
            zmax=1,
        ),
        row=1,
        col=2,
    )

    fig.update_layout(
        template=plotly_template,
        height=600,
        plot_bgcolor=bg_color,
        paper_bgcolor=bg_color,
        margin=dict(l=10, r=10, t=80, b=80),
        legend=dict(groupclick="toggleitem"),
        title="HeartKit: Foundation Demo",
    )
    fig.write_html(params.job_dir / "demo.html", include_plotlyjs="cdn", full_html=True)
    logger.debug(f"Report saved to {params.job_dir / 'demo.html'}")

    if params.display_report:
        fig.show()
