import numpy as np
import pydantic_argparse
import tensorflow as tf
from rich.console import Console
from sklearn.metrics import f1_score

from . import datasets as ds
from .metrics import confusion_matrix_plot, roc_auc_plot
from .models.utils import get_predicted_threshold_indices, get_strategy, load_model
from .types import EcgTestParams
from .utils import set_random_seed, setup_logger

console = Console()
logger = setup_logger(__name__)


def evaluate_model(params: EcgTestParams):
    """Test model command. This evaluates a trained network on the given task and dataset.

    Args:
        params (EcgTestParams): Testing/evaluation parameters
    """
    params.seed = set_random_seed(params.seed)
    logger.info(f"Random seed {params.seed}")

    logger.info("Loading test dataset")
    test_ds = ds.load_test_dataset(
        ds_path=str(params.ds_path),
        task=params.task,
        frame_size=params.frame_size,
        test_patients=params.test_patients,
        test_pt_samples=params.samples_per_patient,
        num_workers=params.data_parallelism,
    )
    with console.status("[bold green] Loading test dataset..."):
        test_x, test_y = next(test_ds.batch(params.test_size).as_numpy_iterator())

    strategy = get_strategy()
    with strategy.scope():
        logger.info("Loading model")
        model = load_model(str(params.model_file))
        model.summary()

        logger.info("Performing inference")
        y_true = test_y
        y_prob = tf.nn.softmax(model.predict(test_x)).numpy()
        y_pred = np.argmax(y_prob, axis=1)

        # Summarize results
        logger.info("Testing Results")
        test_acc = np.sum(y_pred == y_true) / len(y_true)
        test_f1 = f1_score(y_true, y_pred, average="macro")
        logger.info(f"[TEST SET] ACC={test_acc:.2%}, F1={test_f1:.2%}")

        # If threshold given, only count predictions above threshold
        if params.threshold is not None:
            y_thresh_idx = get_predicted_threshold_indices(
                y_prob, y_pred, params.threshold
            )
            drop_perc = 1 - len(y_thresh_idx) / len(y_true)
            y_prob = y_prob[y_thresh_idx]
            y_pred = y_pred[y_thresh_idx]
            y_true = y_true[y_thresh_idx]
            test_acc = np.sum(y_pred == y_true) / len(y_true)
            test_f1 = f1_score(y_true, y_pred, average="macro")
            logger.info(
                f"[TEST SET] ACC={test_acc:.2%}, F1={test_f1:.2%}, THRESH={params.threshold:0.2%}, DROP={drop_perc:.2%}"
            )
        # END IF

        class_names = ds.get_class_names(params.task)
        confusion_matrix_plot(
            y_true,
            y_pred,
            labels=class_names,
            save_path=str(params.job_dir / "confusion_matrix_test.png"),
        )
        if len(class_names) == 2:
            roc_auc_plot(
                y_true,
                y_prob[:, 1],
                labels=class_names,
                save_path=str(params.job_dir / "roc_auc_test.png"),
            )
        # END IF
    # END WITH


def create_parser():
    """Create CLI argument parser
    Returns:
        ArgumentParser: Arg parser
    """
    return pydantic_argparse.ArgumentParser(
        model=EcgTestParams,
        prog="Heart test command",
        description="Test heart model",
    )


if __name__ == "__main__":
    parser = create_parser()
    evaluate_model(parser.parse_typed_args())
