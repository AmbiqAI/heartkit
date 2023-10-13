import tempfile

import tensorflow as tf


def load_model(model_path: str) -> tf.keras.Model:
    """Loads a TF model stored either remotely or locally.
    NOTE: Currently only WANDB and local files are supported.

    Args:
        model_path (str): Source path
            WANDB: wandb://[[entity/]project/]collectionName:[alias]
            FILE: file://path/to/model.tf
            S3: S3://bucket/prefix/model.tf
    Returns:
        tf.keras.Model: Model
    """
    # Stored as WANDB artifact (assumes user is authenticated)
    if model_path.startswith("wandb://"):
        # pylint: disable=C0415
        import wandb  # lazy import wandb

        api = wandb.Api()
        model_path = model_path.removeprefix("wandb://")
        artifact = api.artifact(model_path, type="model")
        with tempfile.TemporaryDirectory() as tmpdirname:
            artifact.download(tmpdirname)
            model = tf.keras.models.load_model(tmpdirname)
        return model
    if model_path.startswith("s3://"):
        raise NotImplementedError("S3 handler not implemented yet")
    # Local file
    model_path = model_path.removeprefix("file://")
    return tf.keras.models.load_model(model_path)


def get_strategy(use_mixed_precision: bool = False) -> tf.distribute.Strategy:
    """Select best distribution strategy.
    Args:
        use_mixed_precision (bool, optional): Use mixed precision on TPU. Defaults to False.
    Returns:
        tf.distribute.Strategy: Strategy
    """
    # Try to detect an available TPU. If none is present, default to MirroredStrategy
    try:
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver.connect()
        strategy = tf.distribute.TPUStrategy(tpu)
        if use_mixed_precision:
            tf.keras.mixed_precision.set_global_policy("mixed_bfloat16")
    except ValueError:
        # MirroredStrategy is best for a single machine with one or multiple GPUs
        strategy = tf.distribute.MirroredStrategy()
    return strategy
