import os
import json

import keras
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import helia_edge as helia
from sklearn.manifold import TSNE

from ...defines import HKTaskParams
from ...datasets import DatasetFactory
from .datasets import load_test_dataset
from ...utils import setup_plotting


def evaluate(params: HKTaskParams):
    """Evaluate model for foundation task using SimCLR

    Args:
        params (HKTaskParams): Task parameters
    """
    os.makedirs(params.job_dir, exist_ok=True)
    logger = helia.utils.setup_logger(__name__, level=params.verbose, file_path=params.job_dir / "test.log")
    logger.debug(f"Creating working directory in {params.job_dir}")

    params.seed = helia.utils.set_random_seed(params.seed)
    logger.debug(f"Random seed {params.seed}")

    datasets = [DatasetFactory.get(ds.name)(**ds.params) for ds in params.datasets]

    # Load validation data
    if params.val_file:
        logger.info(f"Loading validation dataset from {params.val_file}")
        test_ds = tf.data.Dataset.load(str(params.val_file))
    else:
        test_ds = load_test_dataset(datasets=datasets, params=params)

    # Grab sets of augmented samples
    test_x1, test_x2 = [], []
    for inputs in test_ds.as_numpy_iterator():
        test_x1.append(inputs[helia.trainers.SimCLRTrainer.AUG_SAMPLES_0])
        test_x2.append(inputs[helia.trainers.SimCLRTrainer.AUG_SAMPLES_1])
    test_x1 = np.concatenate(test_x1)
    test_x2 = np.concatenate(test_x2)

    logger.debug("Loading model")
    model = helia.models.load_model(params.model_file)
    flops = helia.metrics.flops.get_flops(model, batch_size=1, fpath=params.job_dir / "model_flops.log")

    model.summary(print_fn=logger.debug)
    logger.debug(f"Model requires {flops / 1e6:0.2f} MFLOPS")

    logger.debug("Performing inference")
    test_y1 = model.predict(test_x1)
    test_y2 = model.predict(test_x2)

    metrics = [
        keras.metrics.CosineSimilarity(name="cos"),
        keras.metrics.MeanSquaredError(name="mse"),
    ]

    setup_plotting()
    rst = helia.metrics.compute_metrics(metrics, test_y1, test_y2)
    rst["flops"] = flops
    rst["parameters"] = model.count_params()
    rst = {k: float(v) for k, v in rst.items()}
    logger.info("[TEST SET] " + ", ".join([f"{k.upper()}={v:.4f}" for k, v in rst.items()]))
    with open(params.job_dir / "metrics.json", "w") as fp:
        json.dump(rst, fp)

    # Compute t-SNE
    logger.debug("Computing t-SNE")
    tsne = TSNE(n_components=2, random_state=0, n_iter=1000, perplexity=75)
    x_tsne = tsne.fit_transform(test_y1)

    # Plot t-SNE in matplotlib
    fig, ax = plt.subplots(1, 1, figsize=(9, 9))
    ax.scatter(x_tsne[:, 0], x_tsne[:, 1], c=x_tsne[:, 0] - x_tsne[:, 1], cmap="viridis")
    fig.suptitle("HK Foundation: t-SNE")
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    fig.savefig(params.job_dir / "tsne.png")

    # cleanup
    keras.utils.clear_session()
    for ds in datasets:
        ds.close()
