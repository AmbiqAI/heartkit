import logging
import os
import shutil

import keras
import numpy as np
import plotly.graph_objects as go
import tensorflow as tf
import tensorflow_model_optimization as tfmot
import wandb
from plotly.subplots import make_subplots
from tqdm import tqdm
from wandb.keras import WandbMetricsLogger, WandbModelCheckpoint

from ... import tflite as tfa
from ...datasets import augment_pipeline
from ...defines import HKDemoParams, HKExportParams, HKTestParams, HKTrainParams
from ...rpc.backends import EvbBackend, PcBackend
from ...utils import env_flag, set_random_seed, setup_logger
from ..task import HKTask
from .defines import get_class_shape, get_feat_shape
from .utils import (
    create_model,
    load_datasets,
    load_test_datasets,
    load_train_datasets,
    prepare,
)

logger = setup_logger(__name__)


class Denoise(HKTask):
    """HeartKit ECG Denoise Task"""

    @staticmethod
    def train(params: HKTrainParams):
        """Train model

        Args:
            params (HKTrainParams): Training parameters
        """

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
                project="heartkit-denoise",
                entity="ambiq",
                dir=params.job_dir,
            )
            wandb.config.update(params.model_dump())
        # END IF

        num_classes = 1
        input_spec = (
            tf.TensorSpec(shape=get_feat_shape(params.frame_size), dtype=tf.float32),
            tf.TensorSpec(shape=get_class_shape(params.frame_size, num_classes), dtype=tf.float32),
        )

        datasets = load_datasets(
            ds_path=params.ds_path,
            frame_size=params.frame_size,
            sampling_rate=params.sampling_rate,
            spec=input_spec,
            class_map=None,
            datasets=params.datasets,
        )
        train_ds, val_ds = load_train_datasets(datasets=datasets, params=params)

        with tfa.get_strategy().scope():
            inputs = keras.Input(shape=input_spec[0].shape, batch_size=None, name="input", dtype=input_spec[0].dtype)
            if params.resume and params.model_file:
                logger.info(f"Loading model from file {params.model_file}")
                model = tfa.load_model(params.model_file)
            else:
                logger.info("Creating model from scratch")
                model = create_model(
                    inputs,
                    num_classes=params.num_classes,
                    architecture=params.architecture,
                )
            # END IF

            flops = tfa.get_flops(model, batch_size=1, fpath=params.job_dir / "model_flops.log")

            if params.lr_cycles > 1:
                scheduler = keras.optimizers.schedules.CosineDecayRestarts(
                    initial_learning_rate=params.lr_rate,
                    first_decay_steps=int(0.1 * params.steps_per_epoch * params.epochs),
                    t_mul=1.65 / (0.1 * params.lr_cycles * (params.lr_cycles - 1)),
                    m_mul=0.4,
                )
            else:
                scheduler = keras.optimizers.schedules.CosineDecay(
                    initial_learning_rate=params.lr_rate, decay_steps=params.steps_per_epoch * params.epochs
                )
            # END IF

            optimizer = keras.optimizers.Adam(scheduler)
            loss = keras.losses.MeanSquaredError()
            metrics = [
                keras.metrics.MeanAbsoluteError(name="mae"),
                keras.metrics.MeanSquaredError(name="mse"),
            ]

            if params.resume and params.weights_file:
                logger.info(f"Hydrating model weights from file {params.weights_file}")
                model.load_weights(params.weights_file)

            if params.model_file is None:
                params.model_file = params.job_dir / "model.keras"

            # Perform QAT if requested
            if params.quantization.enabled and params.quantization.qat:
                logger.info("Performing QAT")
                model = tfmot.quantization.keras.quantize_model(model, quantized_layer_name_prefix="q_")
            # END IF

            model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
            model(inputs)
            model.summary(print_fn=logger.info)
            logger.info(f"Model requires {flops/1e6:0.2f} MFLOPS")

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
                    filepath=params.model_file,
                    monitor=f"val_{params.val_metric}",
                    save_best_only=True,
                    save_weights_only=False,
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

            # Get full validation results
            keras.models.load_model(params.model_file)
            logger.info("Performing full validation")
        # END WITH

    @staticmethod
    def evaluate(params: HKTestParams):
        """Evaluate model

        Args:
            params (HKTestParams): Evaluation parameters
        """

        params.seed = set_random_seed(params.seed)
        logger.info(f"Random seed {params.seed}")

        os.makedirs(params.job_dir, exist_ok=True)
        logger.info(f"Creating working directory in {params.job_dir}")

        handler = logging.FileHandler(params.job_dir / "test.log", mode="w")
        handler.setLevel(logging.INFO)
        logger.addHandler(handler)

        num_classes = 1
        input_spec = (
            tf.TensorSpec(shape=get_feat_shape(params.frame_size), dtype=tf.float32),
            tf.TensorSpec(shape=get_class_shape(params.frame_size, num_classes), dtype=tf.float32),
        )

        datasets = load_datasets(
            ds_path=params.ds_path,
            frame_size=params.frame_size,
            sampling_rate=params.sampling_rate,
            spec=input_spec,
            class_map=None,
            datasets=params.datasets,
        )
        test_x, test_y = load_test_datasets(datasets=datasets, params=params)

        with tfmot.quantization.keras.quantize_scope():
            logger.info("Loading model")
            model = tfa.load_model(params.model_file)
            flops = tfa.get_flops(model, batch_size=1, fpath=params.job_dir / "model_flops.log")

            model.summary(print_fn=logger.info)
            logger.info(f"Model requires {flops/1e6:0.2f} MFLOPS")

            logger.info("Performing inference")
            y_true = test_y
            y_prob = model.predict(test_x)
            y_pred = y_prob
        # END WITH

        # Summarize results
        logger.info("Testing Results")
        test_mae = np.mean(np.abs(y_true - y_pred))
        test_rmse = np.sqrt(np.mean(np.square(y_true - y_pred)))
        logger.info(f"[TEST SET] MAE={test_mae:.2%}, RMSE={test_rmse:.2%}")

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

        num_classes = 1
        input_spec = (
            tf.TensorSpec(shape=get_feat_shape(params.frame_size), dtype=tf.float32),
            tf.TensorSpec(shape=get_class_shape(params.frame_size, num_classes), dtype=tf.float32),
        )

        datasets = load_datasets(
            ds_path=params.ds_path,
            frame_size=params.frame_size,
            sampling_rate=params.sampling_rate,
            spec=input_spec,
            class_map=None,
            datasets=params.datasets,
        )
        test_x, test_y = load_test_datasets(datasets=datasets, params=params)

        # Load model and set fixed batch size of 1
        logger.info("Loading trained model")
        with tfmot.quantization.keras.quantize_scope():
            model = tfa.load_model(params.model_file)

        inputs = keras.Input(shape=input_spec[0].shape, batch_size=1, name="input", dtype=input_spec[0].dtype)
        model(inputs)  # Build model with fixed batch size of 1

        flops = tfa.get_flops(model, batch_size=1, fpath=params.job_dir / "model_flops.log")
        model.summary(print_fn=logger.info)
        logger.info(f"Model requires {flops/1e6:0.2f} MFLOPS")

        logger.info(f"Converting model to TFLite (quantization={params.quantization.enabled})")
        if params.quantization.enabled:
            _, quant_df = tfa.debug_quant_tflite(
                model=model,
                test_x=test_x,
                input_type=params.quantization.input_type,
                output_type=params.quantization.output_type,
                supported_ops=params.quantization.supported_ops,
            )
            quant_df.to_csv(params.job_dir / "quant.csv")

        tflite_model = tfa.convert_tflite(
            model=model,
            quantize=params.quantization.enabled,
            test_x=test_x,
            input_type=params.quantization.input_type,
            output_type=params.quantization.output_type,
            supported_ops=params.quantization.supported_ops,
        )

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

        # Verify TFLite results match TF results on example data
        logger.info("Validating model results")
        y_true = test_y
        y_pred_tf = model.predict(test_x)
        y_pred_tfl = tfa.predict_tflite(model_content=tflite_model, test_x=test_x)

        tf_mae = np.mean(np.abs(y_true - y_pred_tf))
        tf_rmse = np.sqrt(np.mean((y_true - y_pred_tf) ** 2))
        logger.info(f"[TF SET] MAE={tf_mae:.2%}, RMSE={tf_rmse:.2%}")

        tfl_mae = np.mean(np.abs(y_true - y_pred_tfl))
        tfl_rmse = np.sqrt(np.mean((y_true - y_pred_tfl) ** 2))
        logger.info(f"[TFL SET] MAE={tfl_mae:.2%}, RMSE={tfl_rmse:.2%}")

        # Check accuracy hit
        tfl_acc_drop = max(0, tf_mae - tfl_mae)
        if params.val_acc_threshold is not None and (1 - tfl_acc_drop) < params.val_acc_threshold:
            logger.warning(f"TFLite accuracy dropped by {tfl_acc_drop:0.2%}")
        elif params.val_acc_threshold:
            logger.info(f"Validation passed ({tfl_acc_drop:0.2%})")

        if params.tflm_file and tflm_model_path != params.tflm_file:
            logger.info(f"Copying TFLM header to {params.tflm_file}")
            shutil.copyfile(tflm_model_path, params.tflm_file)

    @staticmethod
    def demo(params: HKDemoParams):
        """Run segmentation demo.

        Args:
            params (HKDemoParams): Demo parameters
        """
        bg_color = "rgba(38,42,50,1.0)"
        primary_color = "#11acd5"
        secondary_color = "#ce6cff"
        tertiary_color = "rgb(234,52,36)"
        # quaternary_color = "rgb(92,201,154)"
        plotly_template = "plotly_dark"

        # Load backend inference engine
        BackendRunner = EvbBackend if params.backend == "evb" else PcBackend
        runner = BackendRunner(params=params)

        num_classes = 1
        input_spec = (
            tf.TensorSpec(shape=get_feat_shape(params.frame_size), dtype=tf.float32),
            tf.TensorSpec(shape=get_class_shape(params.frame_size, num_classes), dtype=tf.float32),
        )

        # Load data
        ds = load_datasets(
            ds_path=params.ds_path,
            frame_size=10 * params.sampling_rate,
            sampling_rate=params.sampling_rate,
            spec=input_spec,
            class_map=None,
            datasets=params.datasets,
        )[0]
        x = next(ds.signal_generator(ds.uniform_patient_generator(patient_ids=ds.get_test_patient_ids(), repeat=False)))

        y_act = x.copy()
        if params.augmentations:
            x = augment_pipeline(x, augmentations=params.augmentations, sample_rate=params.sampling_rate)
        x = prepare(x, sample_rate=params.sampling_rate, preprocesses=params.preprocesses)
        y_act = prepare(y_act, sample_rate=params.sampling_rate, preprocesses=params.preprocesses)

        # Run inference
        runner.open()
        logger.info("Running inference")
        y_pred = np.zeros(x.size, dtype=np.float32)
        for i in tqdm(range(0, x.size, params.frame_size), desc="Inference"):
            if i + params.frame_size > x.size:
                start, stop = x.size - params.frame_size, x.size
            else:
                start, stop = i, i + params.frame_size
            xx = x[start:stop]
            runner.set_inputs(xx)
            runner.perform_inference()
            yy = runner.get_outputs()
            y_pred[start:stop] = yy.flatten()
        # END FOR
        runner.close()

        # Report
        logger.info("Generating report")
        ts = np.arange(0, x.size) / params.sampling_rate

        fig = make_subplots(
            rows=1,
            cols=3,
            specs=[
                [{"colspan": 3, "type": "xy", "secondary_y": True}, None, None],
            ],
            subplot_titles=("ECG Plot",),
            horizontal_spacing=0.1,
            vertical_spacing=0.2,
        )

        fig.add_trace(
            go.Scatter(
                x=ts,
                y=x,
                name="ECG raw",
                mode="lines",
                line=dict(color=primary_color, width=2),
            ),
            row=1,
            col=1,
            secondary_y=False,
        )

        fig.add_trace(
            go.Scatter(
                x=ts,
                y=y_pred,
                name="ECG clean",
                mode="lines",
                line=dict(color=secondary_color, width=2),
            ),
            row=1,
            col=1,
            secondary_y=True,
        )

        fig.add_trace(
            go.Scatter(
                x=ts,
                y=y_act,
                name="ECG ideal",
                mode="lines",
                line=dict(color=tertiary_color, width=2),
            ),
            row=1,
            col=1,
            secondary_y=False,
        )

        fig.update_xaxes(title_text="Time (s)", row=1, col=1)
        fig.update_yaxes(title_text="ECG", row=1, col=1)

        fig.update_layout(
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            template=plotly_template,
            height=800,
            plot_bgcolor=bg_color,
            paper_bgcolor=bg_color,
            margin=dict(l=10, r=10, t=80, b=80),
            title="HeartKit: ECG Denoise Demo",
        )

        fig.write_html(params.job_dir / "demo.html", include_plotlyjs="cdn", full_html=False)
        fig.show()

        logger.info(f"Report saved to {params.job_dir / 'demo.html'}")
