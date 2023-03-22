import random

import numpy as np
import tensorflow as tf

from neuralspot.tflite.model import load_model

from .datasets import IcentiaDataset
from .datasets.preprocess import preprocess_signal
from .defines import HeartDemoParams, HeartTask
from .hrv import compute_hrv


def demo(params: HeartDemoParams):
    """Pure Python demo."""
    ds_path = ""
    arrhythmia_frame_size = 1000
    arrhythmia_threshold = 0.9
    sampling_rate = 250
    segmentation_threshold = 0.9

    data_len = 20 * sampling_rate
    seg_len = 624
    arr_len = 1000
    seg_olp = 50
    beat_len = 200

    seg_model = load_model("")
    arr_model = load_model("")
    beat_model = load_model("")
    # hrv_model = load_model("")

    # Use Icentia dataset
    ds = IcentiaDataset(
        ds_path=ds_path,
        task=HeartTask.arrhythmia,
        frame_size=arrhythmia_frame_size,
        target_rate=sampling_rate,
    )
    pt_gen = ds.uniform_patient_generator(patient_ids=[1], repeat=False, shuffle=False)
    data = None
    for _, segments in pt_gen:
        seg_key = random.choice(list(segments.keys()))
        frame_start = random.randint(
            5 * sampling_rate, segments[seg_key]["data"].size - 2 * data_len
        )
        frame_end = frame_start + data_len
        data = segments[seg_key]["data"][frame_start:frame_end]
        break

    # Pre-process
    data = preprocess_signal(data=data, sample_rate=sampling_rate)

    # Apply arrhythmia model
    arr_labels = np.zeros((data_len,))
    for i in range(0, data_len - arr_len + 1, arr_len):
        test_x = np.expand_dims(np.expand_dims(data[i : i + arr_len], 0), 0)
        y_prob = tf.nn.softmax(arr_model.predict(test_x)).numpy()
        y_pred = y_prob[0][1] > arrhythmia_threshold
        arr_labels[i : i + arr_len] = y_pred
    # END FOR

    arrhythmia_detected = np.any(arr_labels)
    if arrhythmia_detected:
        print("Arrhythmia onset detected")

    # Apply segmentation model
    seg_mask = np.zeros((data_len,))
    qrs_mask = np.zeros((data_len,))
    for i in range(0, data_len - seg_len + 1, seg_len - 2 * seg_olp):
        test_x = data[i : i + seg_len]
        y_prob = tf.nn.softmax(seg_model.predict(test_x)).numpy()
        y_pred = np.argmax(y_prob, axis=2)
        seg_mask[i + seg_olp : i + seg_len - seg_olp] = y_pred[0, seg_olp:-seg_olp]
        qrs_mask[i + seg_olp : i + seg_len - seg_olp] = np.where(
            y_prob[0, seg_olp:-seg_olp, 2] > segmentation_threshold, 1, 0
        )
    # END FOR

    # Dont apply additional models if arrhythmia detected

    # Apply HRV model and extract R peaks
    hr, rr, rpeaks = compute_hrv(data, seg_mask, sampling_rate)
    print(hr, rr)
    blabels = -1 * np.ones_like(rpeaks)
    for i in range(1, len(rpeaks) - 1):
        prev_start = rpeaks[i - 1] - int(0.5 * beat_len)
        curr_start = rpeaks[i] - int(0.5 * beat_len)
        next_start = rpeaks[i + 1] - int(0.5 * beat_len)
        test_x = np.hstack(
            (
                data[prev_start : prev_start + beat_len],
                data[curr_start : curr_start + beat_len],
                data[next_start : next_start + beat_len],
            )
        )
        y_prob = tf.nn.softmax(beat_model.predict(test_x)).numpy()
        y_pred = np.argmax(y_prob, axis=1)
        blabels[i] = y_pred[0]
    # END FOR

    # Plot ECG data
    # -> Color-coded horizontal lines if arrhythmia present
    # -> Color-coded horizontal lines for segmentation labels
    # -> Color-coded Vertical lines for beat labels

    # Plot Poincare of RR intervals
    # -> Color-code marker as arrhythmia, pac, pvc, normal
