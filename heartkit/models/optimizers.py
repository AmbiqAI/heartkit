import sys

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
