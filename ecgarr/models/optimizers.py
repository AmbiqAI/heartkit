import sys

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

if sys.platform == "darwin":
    Adam = tf.keras.optimizers.legacy.Adam
else:
    Adam = tf.keras.optimizers.Adam


def Ranger():
    """Create Ranger optimizer."""
    radam = tfa.optimizers.RectifiedAdam()
    ranger = tfa.optimizers.Lookahead(radam, sync_period=6, slow_step_size=0.5)
    return ranger


def lr_warmup_cosine_decay(
    global_step: int,
    warmup_steps: int,
    hold: int = 0,
    total_steps: int = 0,
    start_lr: float = 0.0,
    target_lr: float = 1e-2,
):
    """Learning rate warmup with cosine decay strategy

    Args:
        global_step (int): Current step
        warmup_steps (int): # steps to warm up
        hold (int): Hold target_lr before applying cosine decay
        total_steps (int): Total steps in dataset
        start_lr (float, optional): Start LR. Defaults to 0.0.
        target_lr (float, optional): Target LR. Defaults to 1e-2.
    """
    # Cosine decay
    learning_rate = (
        0.5
        * target_lr
        * (
            1
            + tf.cos(
                tf.constant(np.pi)
                * tf.cast(global_step - warmup_steps - hold, tf.float32)
                / float(total_steps - warmup_steps - hold)
            )
        )
    )

    warmup_lr = tf.cast(target_lr * (global_step / warmup_steps), tf.float32)
    target_lr = tf.cast(target_lr, tf.float32)

    if hold > 0:
        learning_rate = tf.where(
            global_step > warmup_steps + hold, learning_rate, target_lr
        )

    learning_rate = tf.where(global_step < warmup_steps, warmup_lr, learning_rate)
    return learning_rate


class WarmUpCosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Cosine decay with warm-up."""

    def __init__(
        self,
        warmup_steps: int,
        total_steps: int,
        hold: int = 0,
        start_lr=0.0,
        target_lr=1e-2,
    ):
        """Keras LR scheduler for warmup with cosine decay strategy

        Args:
            warmup_steps (int): # steps to warm up
            total_steps (int): Total steps in dataset
            hold (int): Hold target_lr before applying cosine decay
            start_lr (float, optional): Start LR. Defaults to 0.0.
            target_lr (float, optional): Target LR. Defaults to 1e-2.
        """
        super().__init__()
        self.start_lr = start_lr
        self.target_lr = target_lr
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.hold = hold

    def __call__(self, step):
        lr = lr_warmup_cosine_decay(
            global_step=step,
            total_steps=self.total_steps,
            warmup_steps=self.warmup_steps,
            start_lr=self.start_lr,
            target_lr=self.target_lr,
            hold=self.hold,
        )
        return tf.where(step > self.total_steps, 0.0, lr, name="learning_rate")

    def get_config(self):
        return dict(
            start_lr=self.start_lr,
            target_lr=self.target_lr,
            warmup_steps=self.warmup_steps,
            total_steps=self.total_steps,
            hold=self.hold,
        )
