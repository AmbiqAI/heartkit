import functools
import logging
import random
from typing import Generator

import keras
import numpy as np
import numpy.typing as npt
import physiokit as pk
import tensorflow as tf
from rich.console import Console

from ...datasets import (
    DatasetFactory,
    HKDataset,
    PatientGenerator,
    PtbxlDataset,
    SampleGenerator,
    augment_pipeline,
    create_dataset_from_data,
    create_interleaved_dataset_from_generator,
    preprocess_pipeline,
    uniform_id_generator,
)
from ...defines import (
    AugmentationParams,
    DatasetParams,
    HKExportParams,
    HKTestParams,
    HKTrainParams,
    ModelArchitecture,
    PreprocessParams,
)
from ...utils import load_pkl

logger = logging.getLogger(__name__)


def preprocess(x: npt.NDArray, y: npt.NDArray, preprocesses: list[PreprocessParams], sample_rate: float):
    return preprocess_pipeline(x, preprocesses=preprocesses, sample_rate=sample_rate), y


def augment(x: npt.NDArray, y: npt.NDArray, augmentations: list[AugmentationParams], sample_rate: float):
    p1, p2 = x.copy(), y.copy
    p1 = augment_pipeline(
        x=p1,
        augmentations=augmentations,
        sample_rate=sample_rate,
    )
    p2 = augment_pipeline(
        x=p2,
        augmentations=augmentations,
        sample_rate=sample_rate,
    )
    return p1, p2


def prepare(
    ds: tf.data.Dataset,
    sample_rate: float,
    preprocesses: list[PreprocessParams],
    augmentations: list[AugmentationParams],
):
    if preprocesses:
        ds = ds.map(lambda x, y: preprocess(x, y, preprocesses, sample_rate))
    # END IF

    if augmentations:
        ds = ds.map(
            lambda x, y: augment(x, y, augmentations, sample_rate), num_parallel_calls=tf.data.experimental.AUTOTUNE
        )
    # END IF


def ptbxl_data_generator(
    self: PtbxlDataset,
    patient_generator: Generator[int, None, None],
    samples_per_patient: int | list[int] = 1,
) -> Generator[tuple[npt.NDArray, npt.NDArray], None, None]:
    """Generate frames and labels using patient generator.
    Currently use two different leads of same subject data as positive pair.
    """

    input_size = int(np.round((self.sampling_rate / self.target_rate) * self.frame_size))
    for pt in patient_generator:
        segment = self.get_patient_data(pt)
        data = segment["data"]
        for _ in range(samples_per_patient):
            leads = random.sample(self.leads, k=2)
            lead_p1 = leads[0]
            lead_p2 = leads[1]
            start_p1 = np.random.randint(0, data.shape[1] - input_size)
            start_p2 = start_p1
            # start_p2 = np.random.randint(0, data.shape[1] - input_size)

            x1 = np.nan_to_num(data[lead_p1, start_p1 : start_p1 + input_size].squeeze()).astype(np.float32)
            x2 = np.nan_to_num(data[lead_p2, start_p2 : start_p2 + input_size].squeeze()).astype(np.float32)

            if self.sampling_rate != self.target_rate:
                x1 = pk.signal.resample_signal(x1, self.sampling_rate, self.target_rate, axis=0)
                x2 = pk.signal.resample_signal(x2, self.sampling_rate, self.target_rate, axis=0)
            # END IF
            yield x1, x2
        # END FOR
    # END FOR


def load_train_datasets(
    datasets: list[HKDataset],
    params: HKTrainParams,
    ds_spec: tuple[tf.TensorSpec, tf.TensorSpec],
) -> tuple[tf.data.Dataset, tf.data.Dataset]:

    id_generator = functools.partial(uniform_id_generator, repeat=True)

    train_datasets = []
    val_datasets = []
    for ds in datasets:
        match ds.name:
            case "ptbxl":
                data_generator = functools.partial(ptbxl_data_generator, ds=ds)
            case _:
                raise ValueError(f"Dataset {ds.name} not supported")

        train_ids, val_ids = ds.get_train_val_patient_ids(
            train_patients=params.train_patients,
            val_patients=params.val_patients,
            train_pt_samples=params.samples_per_patient,
            val_pt_samples=params.val_samples_per_patient,
        )

        # TODO: Filter

        train_ds, val_ds = ds.load_train_datasets(
            train_patients=train_ids,
            val_patients=val_ids,
            train_pt_samples=params.samples_per_patient,
            val_pt_samples=params.val_samples_per_patient,
            val_file=params.val_file,
            val_size=params.val_size,
            preprocess=preprocess,
            num_workers=params.data_parallelism,
        )
        # Create TF datasets
        train_ds = create_interleaved_dataset_from_generator(
            data_generator=data_generator,
            id_generator=id_generator,
            ids=train_ids,
            spec=ds_spec,
            buffer_size=params.buffer_size,
            batch_size=params.batch_size,
            prefetch_size=-1,
            preprocess=preprocess,
            num_workers=params.data_parallelism,
        )

        if ds.cachable and params.val_file and params.val_file.is_file():
            logger.info(f"Loading validation data from file {params.val_file}")
            val = load_pkl(params.val_file)
            val_ds = create_dataset_from_data(val["x"], val["y"], ds_spec)

        else:
            val_ds = create_interleaved_dataset_from_generator(
                data_generator=data_generator,
                id_generator=id_generator,
                ids=val_ids,
                spec=ds_spec,
                buffer_size=params.buffer_size,
                batch_size=params.batch_size,
                prefetch_size=-1,
                preprocess=preprocess,
                num_workers=params.data_parallelism,
            )

        train_ds, val_ds = ds.load_train_datasets(
            train_patients=params.train_patients,
            val_patients=params.val_patients,
            train_pt_samples=params.samples_per_patient,
            val_pt_samples=params.val_samples_per_patient,
            val_file=params.val_file,
            val_size=params.val_size,
            preprocess=preprocess,
            num_workers=params.data_parallelism,
        )
        train_datasets.append(train_ds)
        val_datasets.append(val_ds)
    # END FOR

    ds_weights = np.array([d.weight for d in params.datasets])
    ds_weights = ds_weights / ds_weights.sum()

    train_ds = tf.data.Dataset.sample_from_datasets(train_datasets, weights=ds_weights)
    val_ds = tf.data.Dataset.sample_from_datasets(val_datasets, weights=ds_weights)

    # Shuffle and batch datasets for training

    val_ds = val_ds.batch(
        batch_size=params.batch_size,
        drop_remainder=True,
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    return train_ds, val_ds
