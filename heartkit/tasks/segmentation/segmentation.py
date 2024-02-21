import logging
import os
import shutil

import keras
import numpy as np
import physiokit as pk
import plotly.graph_objects as go
import sklearn.utils
import tensorflow as tf
import tensorflow_model_optimization as tfmot
import wandb
from plotly.subplots import make_subplots
from tqdm import tqdm
from wandb.keras import WandbMetricsLogger, WandbModelCheckpoint

from ... import tflite as tfa
from ...defines import HKDemoParams, HKExportParams, HKTestParams, HKTrainParams
from ...metrics import compute_iou, confusion_matrix_plot, f1_score
from ...rpc.backends import EvbBackend, PcBackend
from ...utils import env_flag, set_random_seed, setup_logger
from ..task import HKTask
from .defines import HeartSegment
from .utils import (
    apply_augmentation_pipeline,
    create_model,
    get_class_shape,
    get_feat_shape,
    load_datasets,
    load_test_datasets,
    load_train_datasets,
    prepare,
)

logger = setup_logger(__name__)


class SegmentationTask(HKTask):
    """HeartKit Segmentation Task"""

    @staticmethod
    def train(params: HKTrainParams):
        """Train model

        Args:
            params (HKTrainParams): Training parameters
        """

        params.finetune = bool(getattr(params, "finetune", False))
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
                project=f"hk-segmentation-{params.num_classes}",
                entity="ambiq",
                dir=params.job_dir,
            )
            wandb.config.update(params.model_dump())
        # END IF

        classes = sorted(list(set(params.class_map.values())))
        class_names = params.class_names or [f"Class {i}" for i in range(params.num_classes)]

        input_spec = (
            tf.TensorSpec(shape=get_feat_shape(params.frame_size), dtype=tf.float32),
            tf.TensorSpec(shape=get_class_shape(params.frame_size, params.num_classes), dtype=tf.int32),
        )

        datasets = load_datasets(
            ds_path=params.ds_path,
            frame_size=params.frame_size,
            sampling_rate=params.sampling_rate,
            spec=input_spec,
            class_map=params.class_map,
            datasets=params.datasets,
        )
        train_ds, val_ds = load_train_datasets(datasets=datasets, params=params)

        test_labels = [y.numpy() for _, y in val_ds]
        y_true = np.argmax(np.concatenate(test_labels).squeeze(), axis=-1).flatten()

        class_weights = 0.25
        if params.class_weights == "balanced":
            class_weights = sklearn.utils.compute_class_weight("balanced", classes=np.array(classes), y=y_true)

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

            # If fine-tune, freeze model encoder weights
            if params.finetune:
                for layer in model.layers:
                    if layer.name.startswith("ENC"):
                        logger.info(f"Freezing {layer.name}")
                        layer.trainable = False
                    # END IF
                # END FOR
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
            loss = keras.losses.CategoricalFocalCrossentropy(
                from_logits=True,
                alpha=class_weights,
            )
            metrics = [
                keras.metrics.CategoricalAccuracy(name="acc"),
                keras.metrics.OneHotIoU(
                    num_classes=params.num_classes,
                    target_class_ids=classes,
                    name="iou",
                ),
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
            y_pred = np.argmax(model.predict(val_ds).squeeze(), axis=-1).flatten()

            cm_path = params.job_dir / "confusion_matrix.png"
            confusion_matrix_plot(y_true, y_pred, labels=class_names, save_path=cm_path, normalize="true")
            if env_flag("WANDB"):
                conf_mat = wandb.plot.confusion_matrix(preds=y_pred, y_true=y_true, class_names=class_names)
                wandb.log({"conf_mat": conf_mat})
            # END IF

            # Summarize results
            test_acc = np.sum(y_pred == y_true) / y_true.size
            test_f1 = f1_score(y_true=y_true, y_pred=y_pred, average="weighted")
            test_iou = compute_iou(y_true, y_pred, average="weighted")
            logger.info(f"[TEST SET] ACC={test_acc:.2%}, F1={test_f1:.2%} IoU={test_iou:0.2%}")
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

        class_names = params.class_names or [f"Class {i}" for i in range(params.num_classes)]

        input_spec = (
            tf.TensorSpec(shape=get_feat_shape(params.frame_size), dtype=tf.float32),
            tf.TensorSpec(shape=get_class_shape(params.frame_size, params.num_classes), dtype=tf.int32),
        )

        datasets = load_datasets(
            ds_path=params.ds_path,
            frame_size=params.frame_size,
            sampling_rate=params.sampling_rate,
            spec=input_spec,
            class_map=params.class_map,
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
            y_true = np.argmax(test_y, axis=-1)
            y_prob = tf.nn.softmax(model.predict(test_x)).numpy()
            y_pred = np.argmax(y_prob, axis=-1)
        # END WITH

        # Summarize results
        logger.info("Testing Results")
        test_acc = np.sum(y_pred == y_true) / y_true.size
        test_iou = compute_iou(y_true, y_pred, average="weighted")
        logger.info(f"[TEST SET] ACC={test_acc:.2%}, IoU={test_iou:.2%}")

        cm_path = params.job_dir / "confusion_matrix_test.png"
        confusion_matrix_plot(
            y_true.flatten(),
            y_pred.flatten(),
            labels=class_names,
            save_path=cm_path,
            normalize="true",
        )

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

        input_spec = (
            tf.TensorSpec(shape=get_feat_shape(params.frame_size), dtype=tf.float32),
            tf.TensorSpec(shape=get_class_shape(params.frame_size, params.num_classes), dtype=tf.int32),
        )

        datasets = load_datasets(
            ds_path=params.ds_path,
            frame_size=params.frame_size,
            sampling_rate=params.sampling_rate,
            spec=input_spec,
            class_map=params.class_map,
            datasets=params.datasets,
        )
        test_x, test_y = load_test_datasets(datasets=datasets, params=params)

        # Load model and set fixed batch size of 1
        logger.info("Loading trained model")
        with tfmot.quantization.keras.quantize_scope():
            model = tfa.load_model(params.model_file)

        inputs = keras.Input(shape=input_spec[0].shape, batch_size=1, name="input", dtype=input_spec[0].dtype)
        outputs = model(inputs)

        if not params.use_logits and not isinstance(model.layers[-1], keras.layers.Softmax):
            outputs = keras.layers.Softmax()(outputs)
            model = keras.Model(inputs, outputs, name=model.name)
            outputs = model(inputs)
        # END IF

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

        # Verify TFLite results match TF results
        logger.info("Validating model results")
        y_true = np.argmax(test_y, axis=-1)
        y_pred_tf = np.argmax(model.predict(test_x), axis=-1)
        y_pred_tfl = np.argmax(tfa.predict_tflite(model_content=tflite_model, test_x=test_x), axis=-1)

        tf_acc = np.sum(y_true == y_pred_tf) / y_true.size
        tf_iou = compute_iou(y_true, y_pred_tf, average="weighted")
        logger.info(f"[TF SET] ACC={tf_acc:.2%}, IoU={tf_iou:.2%}")

        tfl_acc = np.sum(y_true == y_pred_tfl) / y_true.size
        tfl_iou = compute_iou(y_true, y_pred_tfl, average="weighted")
        logger.info(f"[TFL SET] ACC={tfl_acc:.2%}, IoU={tfl_iou:.2%}")

        # Check accuracy hit
        tfl_acc_drop = max(0, tf_acc - tfl_acc)
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
        quaternary_color = "rgb(92,201,154)"
        plotly_template = "plotly_dark"

        # Load backend inference engine
        BackendRunner = EvbBackend if params.backend == "evb" else PcBackend
        runner = BackendRunner(params=params)

        classes = sorted(list(set(params.class_map.values())))
        class_names = params.class_names or [f"Class {i}" for i in range(params.num_classes)]

        input_spec = (
            tf.TensorSpec(shape=get_feat_shape(params.frame_size), dtype=tf.float32),
            tf.TensorSpec(shape=get_class_shape(params.frame_size, params.num_classes), dtype=tf.int32),
        )

        # Load data
        ds = load_datasets(
            ds_path=params.ds_path,
            frame_size=10 * params.sampling_rate,
            sampling_rate=params.sampling_rate,
            spec=input_spec,
            class_map=params.class_map,
            datasets=params.datasets,
        )[0]
        x = next(ds.signal_generator(ds.uniform_patient_generator(patient_ids=ds.get_test_patient_ids(), repeat=False)))

        # Run inference
        runner.open()
        logger.info("Running inference")
        y_pred = np.zeros(x.size, dtype=np.int32)
        for i in tqdm(range(0, x.size, params.frame_size), desc="Inference"):
            if i + params.frame_size > x.size:
                start, stop = x.size - params.frame_size, x.size
            else:
                start, stop = i, i + params.frame_size
            xx = x[start:stop]
            yy = np.zeros(shape=get_class_shape(params.frame_size, params.num_classes), dtype=np.int32)
            xx, yy = apply_augmentation_pipeline(
                xx,
                yy,
                frame_size=params.frame_size,
                sample_rate=params.sampling_rate,
                class_map=params.class_map,
                augmentations=params.augmentations,
            )
            xx = prepare(xx, sample_rate=params.sampling_rate, preprocesses=params.preprocesses)
            runner.set_inputs(xx)
            runner.perform_inference()
            yy = runner.get_outputs()
            y_pred[start:stop] = np.argmax(yy, axis=-1).flatten()
        # END FOR
        runner.close()

        # Report
        logger.info("Generating report")
        ts = np.arange(0, x.size) / params.sampling_rate

        fig = make_subplots(
            rows=2,
            cols=3,
            specs=[
                [{"colspan": 3, "type": "xy", "secondary_y": True}, None, None],
                [{"type": "xy"}, {"type": "bar"}, {"type": "table"}],
            ],
            subplot_titles=("ECG Plot", "IBI Poincare Plot", "HRV Frequency Bands"),
            horizontal_spacing=0.1,
            vertical_spacing=0.2,
        )

        # Extract R peaks from QRS segments
        pred_bounds = np.concatenate(([0], np.diff(y_pred).nonzero()[0] + 1, [y_pred.size - 1]))
        peaks = []
        for i in range(1, len(pred_bounds)):
            start, stop = pred_bounds[i - 1], pred_bounds[i]
            duration = 1000 * (stop - start) / params.sampling_rate
            if y_pred[start] == params.class_map.get(HeartSegment.qrs, -1) and (duration > 20):
                peaks.append(start + np.argmax(np.abs(x[start:stop])))
        peaks = np.array(peaks)

        band_names = ["VLF", "LF", "HF", "VHF"]
        bands = [(0.0033, 0.04), (0.04, 0.15), (0.15, 0.4), (0.4, 0.5)]

        # Compute R-R intervals
        rri = pk.ecg.compute_rr_intervals(peaks)
        mask = pk.ecg.filter_rr_intervals(rri, sample_rate=params.sampling_rate)
        rri_ms = 1000 * rri / params.sampling_rate
        # Compute metrics
        if (rri.size <= 2) or (mask.sum() / mask.size > 0.80):
            logger.warning("High percentage of RR intervals were filtered out")
            hr_bpm = 0
            hrv_td = pk.hrv.HrvTimeMetrics()
            hrv_fd = pk.hrv.HrvFrequencyMetrics(bands=[pk.hrv.HrvFrequencyBandMetrics() for b in bands])
        else:
            hr_bpm = 60 / (np.nanmean(rri[mask == 0]) / params.sampling_rate)
            hrv_td = pk.hrv.compute_hrv_time(rri[mask == 0], sample_rate=params.sampling_rate)
            hrv_fd = pk.hrv.compute_hrv_frequency(
                peaks[mask == 0], rri[mask == 0], bands=bands, sample_rate=params.sampling_rate
            )
        # END IF

        fig.add_trace(
            go.Scatter(
                x=ts,
                y=x,
                name="ECG",
                mode="lines",
                line=dict(color=primary_color, width=2),
            ),
            row=1,
            col=1,
            secondary_y=False,
        )

        for i, peak in enumerate(peaks):
            color = "red" if mask[i] else "white"
            fig.add_vline(
                x=ts[peak],
                line_width=1,
                line_dash="dash",
                line_color=color,
                annotation={"text": "R-Peak", "textangle": -90, "font_color": color},
                row=1,
                col=1,
                secondary_y=False,
            )
        fig.update_xaxes(title_text="Time (s)", row=1, col=1)
        fig.update_yaxes(title_text="ECG", row=1, col=1)

        for i, label in enumerate(classes):
            if label <= 0:
                continue
            fig.add_trace(
                go.Scatter(
                    x=ts,
                    y=np.where(y_pred == label, x, np.nan),
                    name=class_names[i],
                    mode="lines",
                    line_width=2,
                ),
                row=1,
                col=1,
                secondary_y=False,
            )
        # END FOR

        fig.add_trace(
            go.Bar(
                x=np.array([b.total_power for b in hrv_fd.bands]) / hrv_fd.total_power,
                y=band_names,
                marker_color=[primary_color, secondary_color, tertiary_color, quaternary_color],
                orientation="h",
                showlegend=False,
            ),
            row=2,
            col=2,
        )
        fig.update_xaxes(title_text="Normalized Power", row=2, col=2)

        fig.add_trace(
            go.Scatter(
                x=rri_ms[:-1],
                y=rri_ms[1:],
                mode="markers",
                marker_size=10,
                showlegend=False,
                marker_color=secondary_color,
            ),
            row=2,
            col=1,
            secondary_y=False,
        )
        if rri_ms.size > 2:
            rr_min, rr_max = np.nanmin(rri_ms) - 20, np.nanmax(rri_ms) + 20
        else:
            rr_min, rr_max = 0, 2000
        fig.add_shape(
            type="line",
            layer="below",
            y0=rr_min,
            y1=rr_max,
            x0=rr_min,
            x1=rr_max,
            line=dict(color="white", width=2),
            row=2,
            col=1,
        )
        fig.update_xaxes(title_text="RRn (ms)", range=[rr_min, rr_max], row=2, col=1)
        fig.update_yaxes(title_text="RRn+1 (ms)", range=[rr_min, rr_max], row=2, col=1)

        fig.add_trace(
            go.Table(
                header=dict(
                    values=["Heart Rate <br> Variability Metric", "<br> Value"],
                    font=dict(size=16, color="white"),
                    height=60,
                    fill_color=primary_color,
                    align=["left"],
                ),
                cells=dict(
                    values=[
                        ["Heart Rate", "NN Mean", "NN St. Dev", "SD RMS"],
                        [
                            f"{hr_bpm:.0f} BPM",
                            f"{hrv_td.mean_nn:.1f} ms",
                            f"{hrv_td.sd_nn:.1f} ms",
                            f"{hrv_td.rms_sd:.1f}",
                        ],
                    ],
                    font=dict(size=14),
                    height=40,
                    fill_color=bg_color,
                    align=["left"],
                ),
            ),
            row=2,
            col=3,
        )

        fig.update_layout(
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            template=plotly_template,
            height=800,
            plot_bgcolor=bg_color,
            paper_bgcolor=bg_color,
            margin=dict(l=10, r=10, t=80, b=80),
            title="HeartKit: Segmentation Demo",
        )

        fig.write_html(params.job_dir / "demo.html", include_plotlyjs="cdn", full_html=False)
        if env_flag("SHOW"):
            fig.show()

        logger.info(f"Report saved to {params.job_dir / 'demo.html'}")
