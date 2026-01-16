import datetime
import random

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from tqdm import tqdm
import helia_edge as helia

from ...defines import HKTaskParams
from ...backends import BackendFactory
from ...datasets import DatasetFactory


logger = helia.utils.setup_logger(__name__)


def demo(params: HKTaskParams):
    """Run diagnostic demo.

    Args:
        params (HKTaskParams): Task parameters
    """

    bg_color = "rgba(38,42,50,1.0)"
    primary_color = "#11acd5"
    secondary_color = "#ce6cff"
    plotly_template = "plotly_dark"

    params.demo_size = params.demo_size or 2 * params.frame_size

    # Load backend inference engine
    runner = BackendFactory.get(params.backend)(params=params)

    # classes = sorted(list(set(params.class_map.values())))
    class_names = params.class_names or [f"Class {i}" for i in range(params.num_classes)]

    feat_shape = (params.frame_size, 1)
    # class_shape = (params.num_classes,)

    # ds_spec = (
    #     tf.TensorSpec(shape=feat_shape, dtype=tf.float32),
    #     tf.TensorSpec(shape=class_shape, dtype=tf.int32),
    # )

    # Load data
    datasets = [DatasetFactory.get(ds.name)(**ds.params) for ds in params.datasets]
    ds = random.choice(datasets)

    ds_gen = ds.signal_generator(
        patient_generator=helia.utils.uniform_id_generator(ds.get_test_patient_ids(), repeat=False),
        frame_size=params.demo_size,
        samples_per_patient=5,
        target_rate=params.sampling_rate,
    )
    x = next(ds_gen)

    # Run inference
    runner.open()
    logger.debug("Running inference")
    y_pred = np.zeros(x.shape[0], dtype=np.int32)
    for i in tqdm(range(0, x.shape[0], params.frame_size), desc="Inference"):
        if i + params.frame_size > x.shape[0]:
            start, stop = x.shape[0] - params.frame_size, x.shape[0]
        else:
            start, stop = i, i + params.frame_size
        # xx = preprocess(x[start:stop], sample_rate=params.sampling_rate, preprocesses=params.preprocesses)
        xx = x[start:stop]
        xx = xx.reshape(feat_shape)
        runner.set_inputs(xx)
        runner.perform_inference()
        yy = runner.get_outputs()
        y_pred[start:stop] = np.argmax(yy, axis=-1).flatten()
    # END FOR
    runner.close()

    # Report
    logger.debug("Generating report")
    tod = datetime.datetime(2025, 5, 24, random.randint(12, 23), 00)
    ts = np.array([tod + datetime.timedelta(seconds=i / params.sampling_rate) for i in range(x.shape[0])])

    pred_bounds = np.concatenate(([0], np.diff(y_pred).nonzero()[0] + 1, [y_pred.size - 1]))

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
            line=dict(color=primary_color, width=2),
            showlegend=False,
        ),
        row=1,
        col=1,
        secondary_y=False,
    )

    for i in range(1, len(pred_bounds)):
        start, stop = pred_bounds[i - 1], pred_bounds[i]
        pred_class = y_pred[start]
        if pred_class <= 0:
            continue
        fig.add_vrect(
            x0=ts[start],
            x1=ts[stop],
            annotation_text=class_names[pred_class],
            fillcolor=secondary_color,
            opacity=0.25,
            line_width=2,
            line_color=secondary_color,
            row=1,
            col=1,
            secondary_y=False,
        )

    fig.update_layout(
        template=plotly_template,
        height=600,
        plot_bgcolor=bg_color,
        paper_bgcolor=bg_color,
        margin=dict(l=10, r=10, t=80, b=80),
        legend=dict(groupclick="toggleitem"),
        title="heartKIT: Diagnostic Demo",
    )

    fig.write_html(params.job_dir / "demo.html", include_plotlyjs="cdn", full_html=True)
    logger.debug(f"Report saved to {params.job_dir / 'demo.html'}")

    if params.display_report:
        fig.show()
