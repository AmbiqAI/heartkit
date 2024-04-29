import logging
import os

import keras
import numpy as np
import tensorflow as tf
import tensorflow_model_optimization as tfmot
import wandb
from wandb.keras import WandbMetricsLogger, WandbModelCheckpoint

from ... import tflite as tfa
from ...defines import HKDemoParams, HKExportParams, HKTestParams, HKTrainParams
from ...utils import env_flag, set_random_seed, setup_logger
from ..task import HKTask
from .utils import (
    SimCLR,
    create_model,
    get_feat_shape,
    load_datasets,
    load_test_datasets,
    load_train_datasets,
)

logger = setup_logger(__name__)


class FoundationTask(HKTask):
    """HeartKit Foundation Task"""

    @staticmethod
    def train(params: HKTrainParams):
        """Train  model

        Args:
            params (HKTrainParams): Training parameters
        """

        params.temperature = float(getattr(params, "temperature", 0.1))

        params.seed = set_random_seed(params.seed)
        logger.info(f"Random seed {params.seed}")

        os.makedirs(params.job_dir, exist_ok=True)
        logger.info(f"Creating working directory in {params.job_dir}")

        handler = logging.FileHandler(params.job_dir / "train.log", mode="w")
        handler.setLevel(logging.INFO)
        logger.addHandler(handler)

        with open(params.job_dir / "train_config.json", "w", encoding="utf-8") as fp:
            fp.write(params.model_dump_json(indent=2))

        if env_flag("WANDB"):
            wandb.init(
                project=f"hk-foundation-{params.num_classes}",
                entity="ambiq",
                dir=params.job_dir,
            )
            wandb.config.update(params.model_dump())
        # END IF

        # Currently we return positive pairs w/o labels
        ds_spec = (
            tf.TensorSpec(shape=get_feat_shape(params.frame_size), dtype=tf.float32),
            tf.TensorSpec(shape=get_feat_shape(params.frame_size), dtype=tf.float32),
        )

        datasets = load_datasets(
            ds_path=params.ds_path,
            frame_size=params.frame_size,
            sampling_rate=params.sampling_rate,
            spec=ds_spec,
            class_map=None,
            datasets=params.datasets,
        )
        train_ds, val_ds = load_train_datasets(datasets=datasets, params=params)

        projection_width = params.num_classes

        encoder_input = keras.Input(shape=get_feat_shape(params.frame_size), dtype=tf.float32)

        # Encoder
        encoder = create_model(
            encoder_input,
            num_classes=None,
            architecture=params.architecture,
        )
        encoder_output = encoder(encoder_input)
        flops = tfa.get_flops(encoder, batch_size=1, fpath=params.job_dir / "encoder_flops.log")
        encoder.summary(print_fn=logger.info)
        logger.info(f"Encoder requires {flops/1e6:0.2f} MFLOPS")

        # Projector
        projector_input = encoder_output
        projector_output = keras.layers.Dense(projection_width, activation="relu6")(projector_input)
        projector_output = keras.layers.Dense(projection_width, activation="relu6")(projector_output)
        projector = keras.Model(inputs=projector_input, outputs=projector_output, name="projector")
        flops = tfa.get_flops(encoder, batch_size=1, fpath=params.job_dir / "projector_flops.log")
        projector.summary(print_fn=logger.info)

        if params.model_file is None:
            params.model_file = params.job_dir / "model.keras"

        model = SimCLR(
            contrastive_augmenter=lambda x: x,
            encoder=encoder,
            projector=projector,
            temperature=params.temperature,
        )

        def get_scheduler():
            if params.lr_cycles > 1:
                return keras.optimizers.schedules.CosineDecayRestarts(
                    initial_learning_rate=params.lr_rate,
                    first_decay_steps=int(0.1 * params.steps_per_epoch * params.epochs),
                    t_mul=1.65 / (0.1 * params.lr_cycles * (params.lr_cycles - 1)),
                    m_mul=0.4,
                )
            return keras.optimizers.schedules.CosineDecay(
                initial_learning_rate=params.lr_rate, decay_steps=params.steps_per_epoch * params.epochs
            )

        model.compile(
            contrastive_optimizer=keras.optimizers.Adam(get_scheduler()),
            probe_optimizer=keras.optimizers.Adam(get_scheduler()),
        )

        ModelCheckpoint = keras.callbacks.ModelCheckpoint
        if env_flag("WANDB"):
            ModelCheckpoint = WandbModelCheckpoint
        model_callbacks = [
            keras.callbacks.EarlyStopping(
                monitor=f"val_{params.val_metric}",
                patience=max(int(0.25 * params.epochs), 1),
                mode="max" if params.val_metric == "f1" else "auto",
                restore_best_weights=True,
            ),
            ModelCheckpoint(
                filepath=str(params.model_file),
                monitor=f"val_{params.val_metric}",
                save_best_only=True,
                mode="max" if params.val_metric == "f1" else "auto",
                verbose=1,
            ),
            keras.callbacks.CSVLogger(params.job_dir / "history.csv"),
        ]
        if env_flag("TENSORBOARD"):
            model_callbacks.append(
                keras.callbacks.TensorBoard(
                    log_dir=params.job_dir,
                    write_steps_per_second=True,
                )
            )
        if env_flag("WANDB"):
            model_callbacks.append(WandbMetricsLogger())

        try:
            model.fit(
                train_ds,
                steps_per_epoch=params.steps_per_epoch,
                verbose=2,
                epochs=params.epochs,
                validation_data=val_ds,
                callbacks=model_callbacks,
            )
        except KeyboardInterrupt:
            logger.warning("Stopping training due to keyboard interrupt")

        logger.info(f"Model saved to {params.model_file}")

    @staticmethod
    def evaluate(params: HKTestParams):
        """Evaluate model

        Args:
            params (HKTestParams): Evaluation parameters
        """
        # Would need encoder along with either projector or classifier to evaluate

        return

    @staticmethod
    def export(params: HKExportParams):
        """Export model

        Args:
            params (HKExportParams): Deployment parameters
        """

        os.makedirs(params.job_dir, exist_ok=True)
        logger.info(f"Creating working directory in {params.job_dir}")

        handler = logging.FileHandler(params.job_dir / "export.log", mode="w")
        handler.setLevel(logging.INFO)
        logger.addHandler(handler)

        tfl_model_path = params.job_dir / "model.tflite"
        tflm_model_path = params.job_dir / "model_buffer.h"

        ds_spec = (
            tf.TensorSpec(shape=get_feat_shape(params.frame_size), dtype=tf.float32),
            tf.TensorSpec(shape=get_feat_shape(params.frame_size), dtype=tf.float32),
        )

        datasets = load_datasets(
            ds_path=params.ds_path,
            frame_size=params.frame_size,
            sampling_rate=params.sampling_rate,
            spec=ds_spec,
            class_map=None,
            datasets=params.datasets,
        )

        test_x, _ = load_test_datasets(datasets=datasets, params=params)

        # Load model and set fixed batch size of 1
        logger.info("Loading trained model")
        with tfmot.quantization.keras.quantize_scope():
            model = tfa.load_model(params.model_file)

        inputs = keras.Input(shape=get_feat_shape(params.frame_size), batch_size=1, dtype=tf.float32)
        model(inputs)

        flops = tfa.get_flops(model, batch_size=1, fpath=params.job_dir / "model_flops.log")
        model.summary(print_fn=logger.info)
        logger.info(f"Model requires {flops/1e6:0.2f} MFLOPS")

        converter = tfa.create_tflite_converter(
            model=model,
            quantize=params.quantization.enabled,
            test_x=test_x,
            input_type=params.quantization.input_type,
            output_type=params.quantization.output_type,
            supported_ops=params.quantization.supported_ops,
            use_concrete=True,
            feat_shape=get_feat_shape(params.frame_size),
        )
        tflite_model = converter.convert()

        # Save TFLite model
        logger.info(f"Saving TFLite model to {tfl_model_path}")
        with open(tfl_model_path, "wb") as fp:
            fp.write(tflite_model)

        # Save TFLM model
        logger.info(f"Saving TFL micro model to {tflm_model_path}")
        tfa.xxd_c_dump(
            src_path=tfl_model_path,
            dst_path=tflm_model_path,
            var_name=params.tflm_var_name,
            chunk_len=20,
            is_header=True,
        )

        y_pred_tf = model.predict(test_x)
        y_pred_tfl = tfa.predict_tflite(model_content=tflite_model, test_x=test_x)
        print(y_pred_tf.shape)

        # Compare error between TF and TFLite outputs
        error = np.abs(y_pred_tf - y_pred_tfl).max()
        logger.info(f"Max error between TF and TFLite outputs: {error}")

    @staticmethod
    def demo(params: HKDemoParams):
        """Run demo for model

        Args:
            params (HKDemoParams): Demo parameters
        """

        return
