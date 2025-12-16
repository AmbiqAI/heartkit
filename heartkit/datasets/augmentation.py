import keras
import numpy as np
import helia_edge as helia

from ..defines import NamedParams
from .nstdb import NstdbNoise


def create_augmentation_layer(augmentation: NamedParams, sampling_rate: int) -> keras.Layer:
    """Create an augmentation layer from a configuration

    Args:
        augmentation (NamedParams): Augmentation configuration
        sampling_rate (int): Sampling rate of the data

    Returns:
        keras.Layer: Augmentation layer

    Example:

    ```python
    import heartkit as hk
    x = keras.random.normal
    layer = hk.datasets.augmentation.create_augmentation_layer(
        hk.NamedParams(name="random_noise", params={"factor": 0.01}),
        sampling_rate=100
    )
    y = layer(x)
    ```
    """
    match augmentation.name:
        case "amplitude_warp":
            return helia.layers.preprocessing.AmplitudeWarp(sample_rate=sampling_rate, **augmentation.params)
        case "augmentation_pipeline":
            return create_augmentation_pipeline(augmentation.params)
        case "random_augmentation":
            return helia.layers.preprocessing.RandomAugmentation1DPipeline(
                layers=[
                    create_augmentation_layer(augmentation, sampling_rate=sampling_rate)
                    for augmentation in [NamedParams(**p) for p in augmentation.params["layers"]]
                ],
                augmentations_per_sample=augmentation.params.get("augmentations_per_sample", 3),
                rate=augmentation.params.get("rate", 1.0),
                batchwise=True,
            )
        case "random_background_noise":
            nstdb = NstdbNoise(target_rate=sampling_rate)
            noises = np.hstack(
                (nstdb.get_noise(noise_type="bw"), nstdb.get_noise(noise_type="ma"), nstdb.get_noise(noise_type="em"))
            )
            noises = noises.astype(np.float32)
            return helia.layers.preprocessing.RandomBackgroundNoises1D(noises=noises, **augmentation.params)
        case "random_sine_wave":
            return helia.layers.preprocessing.RandomSineWave(**augmentation.params, sample_rate=sampling_rate)
        case "random_cutout":
            return helia.layers.preprocessing.RandomCutout1D(**augmentation.params)
        case "random_noise":
            return helia.layers.preprocessing.RandomGaussianNoise1D(**augmentation.params)
        case "random_noise_distortion":
            return helia.layers.preprocessing.RandomNoiseDistortion1D(sample_rate=sampling_rate, **augmentation.params)
        case "resizing":
            return helia.layers.preprocessing.Resizing1D(**augmentation.params)
        case "sine_wave":
            return helia.layers.preprocessing.AddSineWave(**augmentation.params)
        case "filter":
            return helia.layers.preprocessing.CascadedBiquadFilter(sample_rate=sampling_rate, **augmentation.params)
        case "layer_norm":
            return helia.layers.preprocessing.LayerNormalization1D(**augmentation.params)
        case _:
            raise ValueError(f"Unknown augmentation '{augmentation.name}'")
    # END MATCH


def create_augmentation_pipeline(
    augmentations: list[NamedParams], sampling_rate: int
) -> helia.layers.preprocessing.AugmentationPipeline:
    """Create an augmentation pipeline from a list of augmentation configurations.

    This is useful when running from a configuration file to hydrate the pipeline.

    Args:
        augmentations (list[NamedParams]): List of augmentation configurations
        sampling_rate (int): Sampling rate of the data

    Returns:
        helia.layers.preprocessing.AugmentationPipeline: Augmentation pipeline

    Example:

    ```python
    import heartkit as hk
    x = keras.random.normal(shape=(256, 1), dtype="float32")

    augmenter = hk.datasets.create_augmentation_pipeline([
        hk.NamedParams(name="random_noise", params={"factor": 0.01}),
        hk.NamedParams(name="random_cutout", params={"factor": 0.01, "cutouts": 2}),
    ], sampling_rate=100)

    y = augmenter(x)
    """
    if not augmentations:
        return keras.layers.Lambda(lambda x: x)
    aug = helia.layers.preprocessing.AugmentationPipeline(
        layers=[create_augmentation_layer(augmentation, sampling_rate=sampling_rate) for augmentation in augmentations]
    )
    return aug
