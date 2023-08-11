from typing import Callable

import numpy as np
import numpy.typing as npt
import tensorflow as tf

from ..utils import setup_logger

logger = setup_logger(__name__)


class CustomCheckpoint(tf.keras.callbacks.Callback):
    """Custom keras callback checkpoint"""

    def __init__(
        self,
        filepath: str,
        data: tuple[npt.NDArray, npt.NDArray],
        score_fn: Callable[[tuple[npt.NDArray, npt.NDArray]], npt.NDArray],
        best: float = -np.Inf,
        save_best_only: bool = False,
        batch_size: bool | None = None,
        verbose: int = 0,
    ):
        """Custom keras callback checkpoint

        Args:
            filepath (str): Save checkpoint filepath
            data (tuple[npt.NDArray, npt.NDArray]): Data
            score_fn (Callable[[tuple[npt.NDArray, npt.NDArray]], npt.NDArray]): Scoring function
            best (float, optional): Current best score. Defaults to -np.Inf.
            save_best_only (bool, optional): Save best checkpoint only. Defaults to False.
            batch_size (bool | None, optional): Batch size. Defaults to None.
            verbose (int, optional): Verbosity. Defaults to 0.
        """
        super().__init__()
        self.filepath = filepath
        self.data = data
        self.score_fn = score_fn
        self.save_best_only = save_best_only
        self.batch_size = batch_size
        self.verbose = verbose
        self.best = best

    def on_epoch_end(self, epoch: int, logs=None):
        logs = logs or {}
        x, y_true = self.data
        y_prob = self.model.predict(x, batch_size=self.batch_size)
        score = self.score_fn(y_true, y_prob)
        logs.update({self.metric_name: score})
        filepath = self.filepath.format(epoch=epoch + 1, **logs)
        if score > self.best:
            logger.debug(
                (
                    f"\nEpoch {epoch+1:05d}: {self.metric_name} ({score:.05f}) improved from {self.best:0.5f}"
                    f"saving model to {self.filepath}"
                )
            )
            self.model.save_weights(filepath, overwrite=True)
            self.best = score
        elif not self.save_best_only:
            logger.debug(
                (
                    f"\nEpoch {epoch+1:05d}: {self.metric_name} ({score:.05f}) did not improve from {self.best:0.5f}"
                    f"saving model to {self.filepath}"
                )
            )
            self.model.save_weights(filepath, overwrite=True)
        else:
            logger.debug(
                f"\nEpoch {epoch+1:05d}: {self.metric_name} ({score:.05f}) did not improve from {self.best:0.5f}"
            )

    @property
    def metric_name(self) -> str:
        """Get metric name

        Returns:
            str: name
        """
        return self.score_fn.__name__
