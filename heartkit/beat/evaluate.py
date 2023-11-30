import logging
import os

import numpy as np
import tensorflow as tf

from .. import tflite as tfa
from ..defines import HeartTestParams
from ..metrics import confusion_matrix_plot, f1_score, roc_auc_plot
from ..models.utils import threshold_predictions
from ..utils import set_random_seed, setup_logger
from .defines import get_class_mapping, get_class_names
from .utils import load_dataset, load_test_dataset

logger = setup_logger(__name__)


def evaluate(params: HeartTestParams):
    """Test beat-level model.

    Args:
        params (HeartTestParams): Testing/evaluation parameters
    """
    params.seed = set_random_seed(params.seed)

    os.makedirs(params.job_dir, exist_ok=True)
    logger.info(f"Creating working directory in {params.job_dir}")

    handler = logging.FileHandler(params.job_dir / "test.log", mode="w")
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)

    logger.info(f"Random seed {params.seed}")

    class_names = get_class_names(params.num_classes)
    class_map = get_class_mapping(params.num_classes)

    ds = load_dataset(
        ds_path=params.ds_path, frame_size=params.frame_size, sampling_rate=params.sampling_rate, class_map=class_map
    )
    test_x, test_y = load_test_dataset(ds, params)

    strategy = tfa.get_strategy()
    with strategy.scope():
        logger.info("Loading model")
        model = tfa.load_model(params.model_file)
        flops = tfa.get_flops(model, batch_size=1, fpath=params.job_dir / "model_flops.log")
        model.summary(print_fn=logger.info)
        logger.info(f"Model requires {flops/1e6:0.2f} MFLOPS")

        logger.info("Performing inference")
        y_true = np.argmax(test_y, axis=1)
        y_prob = tf.nn.softmax(model.predict(test_x)).numpy()
        y_pred = np.argmax(y_prob, axis=1)

        # Summarize results
        logger.info("Testing Results")
        test_acc = np.sum(y_pred == y_true) / len(y_true)
        test_f1 = f1_score(y_true, y_pred, average="weighted")
        logger.info(f"[TEST SET] ACC={test_acc:.2%}, F1={test_f1:.2%}")

        # If threshold given, only count predictions above threshold
        if params.threshold:
            numel = len(y_true)
            y_prob, y_pred, y_true = threshold_predictions(y_prob, y_pred, y_true, params.threshold)
            drop_perc = 1 - len(y_true) / numel
            test_acc = np.sum(y_pred == y_true) / len(y_true)
            test_f1 = f1_score(y_true, y_pred, average="weighted")
            logger.info(f"[TEST SET] THRESH={params.threshold:0.2%}, DROP={drop_perc:.2%}")
            logger.info(f"[TEST SET] ACC={test_acc:.2%}, F1={test_f1:.2%}")
        # END IF

        cm_path = params.job_dir / "confusion_matrix_test.png"
        confusion_matrix_plot(y_true, y_pred, labels=class_names, save_path=cm_path, normalize="true")
        if len(class_names) == 2:
            roc_path = params.job_dir / "roc_auc_test.png"
            roc_auc_plot(y_true, y_prob[:, 1], labels=class_names, save_path=roc_path)
        # END IF
    # END WITH
