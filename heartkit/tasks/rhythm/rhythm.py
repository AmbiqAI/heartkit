import datetime
import logging
import os
import random
import shutil

import keras
import numpy as np
import plotly.graph_objects as go
import sklearn.utils
import tensorflow as tf
import tensorflow_model_optimization as tfmot
import wandb
from plotly.subplots import make_subplots
from sklearn.metrics import f1_score
from tqdm import tqdm
from wandb.keras import WandbMetricsLogger, WandbModelCheckpoint

from ... import tflite as tfa
from ...defines import HKDemoParams, HKExportParams, HKTestParams, HKTrainParams
from ...metrics import confusion_matrix_plot, px_plot_confusion_matrix, roc_auc_plot
from ...models.utils import threshold_predictions
from ...rpc import BackendFactory
from ...utils import env_flag, set_random_seed, setup_logger
from ..task import HKTask
from .utils import (
    create_model,
    get_class_shape,
    get_feat_shape,
    load_datasets,
    load_test_datasets,
    load_train_datasets,
    prepare,
)

logger = setup_logger(__name__)


class RhythmTask(HKTask):
    """HeartKit Rhythm Task"""

    @staticmethod
    def train(params: HKTrainParams):
        """Train  model

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
                project=f"hk-rhythm-{params.num_classes}",
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

        test_labels = [label.numpy() for _, label in val_ds]
        y_true = np.argmax(np.concatenate(test_labels), axis=-1)

        class_weights = 0.25
        if params.class_weights == "balanced":
            class_weights = sklearn.utils.compute_class_weight("balanced", classes=np.array(classes), y=y_true)
            class_weights = (class_weights + class_weights.mean()) / 2  # Smooth out
        # END IF
        logger.info(f"Class weights: {class_weights}")

        with tfa.get_strategy().scope():
            inputs = keras.Input(shape=input_spec[0].shape, batch_size=None, name="input", dtype=input_spec[0].dtype)

            # Load existing model
            if params.model_file and params.resume:
                prev_model = tfa.load_model(params.model_file)
                outputs = prev_model(inputs)
                # Stack new layers on top of existing model
                if params.architecture:
                    model = create_model(
                        outputs,
                        num_classes=params.num_classes,
                        architecture=params.architecture,
                    )
                    outputs = model(outputs)
                    model = keras.Model(inputs, outputs, name=model.name)
                # Replace model with existing model
                else:
                    model = prev_model
                # END IF
                params.model_file = None  # Dont overwrite existing model
            else:
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
            loss = keras.losses.CategoricalFocalCrossentropy(from_logits=True, alpha=class_weights)
            metrics = [
                keras.metrics.CategoricalAccuracy(name="acc"),
                tfa.MultiF1Score(name="f1", average="weighted"),
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

            # Get full validation results
            keras.models.load_model(params.model_file)
            logger.info("Performing full validation")
            y_pred = np.argmax(model.predict(val_ds), axis=-1)

            cm_path = params.job_dir / "confusion_matrix.png"
            confusion_matrix_plot(y_true, y_pred, labels=class_names, save_path=cm_path, normalize="true")
            if env_flag("WANDB"):
                conf_mat = wandb.plot.confusion_matrix(preds=y_pred, y_true=y_true, class_names=class_names)
                wandb.log({"conf_mat": conf_mat})
            # END IF

            # Summarize results
            test_acc = np.sum(y_pred == y_true) / len(y_true)
            test_f1 = f1_score(y_true, y_pred, average="weighted")
            logger.info(f"[VAL SET] ACC={test_acc:.2%}, F1={test_f1:.2%}")
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
            class_map=params.class_map,
            spec=input_spec,
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
        test_acc = np.sum(y_pred == y_true) / len(y_true)
        test_f1 = f1_score(y_true, y_pred, average="weighted")
        logger.info(f"[TEST SET] ACC={test_acc:.2%}, F1={test_f1:.2%}")

        if params.num_classes == 2:
            roc_path = params.job_dir / "roc_auc_test.png"
            roc_auc_plot(y_true, y_prob[:, 1], labels=class_names, save_path=roc_path)
        # END IF

        # If threshold given, only count predictions above threshold
        if params.threshold:
            prev_numel = len(y_true)
            y_prob, y_pred, y_true = threshold_predictions(y_prob, y_pred, y_true, params.threshold)
            drop_perc = 1 - len(y_true) / prev_numel
            test_acc = np.sum(y_pred == y_true) / len(y_true)
            test_f1 = f1_score(y_true, y_pred, average="weighted")
            logger.info(f"[TEST SET] THRESH={params.threshold:0.2%}, DROP={drop_perc:.2%}")
            logger.info(f"[TEST SET] ACC={test_acc:.2%}, F1={test_f1:.2%}")
        # END IF
        print(y_true.shape, y_pred.shape)
        cm_path = params.job_dir / "confusion_matrix_test.png"
        confusion_matrix_plot(y_true, y_pred, labels=class_names, save_path=cm_path, normalize="true")
        px_plot_confusion_matrix(
            y_true, y_pred, labels=class_names, save_path=cm_path.with_suffix(".html"), normalize="true"
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

        # classes = sorted(list(set(params.class_map.values())))
        # class_names = params.class_names or [f"Class {i}" for i in range(params.num_classes)]

        input_spec = (
            tf.TensorSpec(shape=get_feat_shape(params.frame_size), dtype=tf.float32),
            tf.TensorSpec(shape=get_class_shape(params.frame_size, params.num_classes), dtype=tf.int32),
        )

        datasets = load_datasets(
            ds_path=params.ds_path,
            frame_size=params.frame_size,
            sampling_rate=params.sampling_rate,
            class_map=params.class_map,
            spec=input_spec,
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

        # if params.quantization.enabled:
        #     _, quant_df = tfa.debug_quant_tflite(
        #        converter=converter
        #     )
        #     quant_df.to_csv(params.job_dir / "quant.csv")
        # # END IF

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
        tf_f1 = f1_score(y_true, y_pred_tf, average="weighted")
        logger.info(f"[TF SET] ACC={tf_acc:.2%}, F1={tf_f1:.2%}")

        tfl_acc = np.sum(y_true == y_pred_tfl) / y_true.size
        tfl_f1 = f1_score(y_true, y_pred_tfl, average="weighted")
        logger.info(f"[TFL SET] ACC={tfl_acc:.2%}, F1={tfl_f1:.2%}")

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
        """Run demo for model

        Args:
            params (HKDemoParams): Demo parameters
        """

        bg_color = "rgba(38,42,50,1.0)"
        primary_color = "#11acd5"
        secondary_color = "#ce6cff"
        plotly_template = "plotly_dark"

        params.demo_size = params.demo_size or 2 * params.frame_size

        # Load backend inference engine
        runner = BackendFactory.create(params.backend, params=params)

        # Load data
        # classes = sorted(list(set(params.class_map.values())))
        class_names = params.class_names or [f"Class {i}" for i in range(params.num_classes)]

        input_spec = (
            tf.TensorSpec(shape=get_feat_shape(params.frame_size), dtype=tf.float32),
            tf.TensorSpec(shape=get_class_shape(params.frame_size, params.num_classes), dtype=tf.int32),
        )

        ds = load_datasets(
            ds_path=params.ds_path,
            frame_size=params.demo_size,
            sampling_rate=params.sampling_rate,
            class_map=params.class_map,
            spec=input_spec,
            datasets=params.datasets,
        )[0]
        x = next(ds.signal_generator(ds.uniform_patient_generator(patient_ids=ds.get_test_patient_ids(), repeat=False)))

        # Run inference
        runner.open()
        logger.info("Running inference")
        y_pred = np.zeros(x.shape[0], dtype=np.int32)
        for i in tqdm(range(0, x.shape[0], params.frame_size), desc="Inference"):
            if i + params.frame_size > x.shape[0]:
                start, stop = x.shape[0] - params.frame_size, x.shape[0]
            else:
                start, stop = i, i + params.frame_size
            xx = prepare(x[start:stop], sample_rate=params.sampling_rate, preprocesses=params.preprocesses)
            runner.set_inputs(xx)
            runner.perform_inference()
            yy = runner.get_outputs()
            y_pred[start:stop] = np.argmax(yy, axis=-1).flatten()
        # END FOR
        runner.close()

        # Report
        logger.info("Generating report")
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
            title="HeartKit: Rhythm Demo",
        )
        fig.write_html(params.job_dir / "demo.html", include_plotlyjs="cdn", full_html=True)
        logger.info(f"Report saved to {params.job_dir / 'demo.html'}")

        if params.display_report:
            fig.show()
