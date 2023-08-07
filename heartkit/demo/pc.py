import time
from typing import Generator

import numpy as np
import numpy.typing as npt
from requests.exceptions import ConnectionError as ReqConnectionError
from requests.exceptions import ConnectTimeout, HTTPError
from rich.console import Console

from neuralspot.tflite.model import load_model

from ..datasets import IcentiaDataset
from ..defines import HeartBeat, HeartRate, HeartSegment
from ..signal import (
    compute_rr_intervals,
    filter_rr_intervals,
    filter_signal,
    find_peaks,
    normalize_signal,
)
from .client import HKRestClient
from .defines import AppState, HeartDemoParams, HeartKitState, HKResult
from .utils import setup_logger

console = Console()
logger = setup_logger(__name__)


class PcHandler:
    """PC Handler. Runs HeartKit models locally"""

    def __init__(self, params: HeartDemoParams) -> None:
        self.params = params
        self.client = HKRestClient(addr=params.rest_address)

        # HeartKit state
        self.hk_state = HeartKitState(
            data_id=0,
            app_state=AppState.IDLE_STATE,
            data=params.frame_size * [0],
            seg_mask=params.frame_size * [0],
            results=HKResult(),
        )

        self._run = False
        self.data_gen = self.create_data_generator()
        self.arr_model = None
        self.seg_model = None
        self.beat_model = None

    def create_data_generator(self) -> Generator[npt.NDArray[np.float32], None, None]:
        """Create data generator

        Returns:
            Generator[npt.NDArray[np.float32], None, None]: Data generator
        """
        ds = IcentiaDataset(
            ds_path=str(self.params.ds_path),
            frame_size=self.params.frame_size,
            target_rate=self.params.sampling_rate,
        )
        pt_gen = ds.uniform_patient_generator(ds.get_train_patient_ids())
        data_gen = ds.signal_generator(pt_gen, samples_per_patient=self.params.samples_per_patient)
        return data_gen

    def load_models(self):
        """Load all models"""
        if self.params.segmentation_model:
            self.seg_model = load_model(self.params.segmentation_model)
        if self.params.arrhythmia_model:
            self.arr_model = load_model(self.params.arrhythmia_model)
        if self.params.beat_model:
            self.beat_model = load_model(self.params.beat_model)

    def preprocess(self, data: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        """Perform pre-processing to data"""
        data = filter_signal(data, lowcut=0.5, highcut=30, order=3, sample_rate=self.params.sampling_rate, axis=0)
        data = normalize_signal(data, eps=0.1, axis=None)
        return data

    def arrhythmia_inference(self, data: npt.NDArray[np.float32], threshold: float = 0.75) -> npt.NDArray[np.uint8]:
        """Apply arrhythmia model to data.

        Args:
            data (npt.NDArray[np.float32]): ECG data
            threshold (float, optional): Confidence threshold. Defaults to 0.75.

        Returns:
            npt.NDArray[np.uint8]: Arrhythmia label
        """
        data_len = data.shape[0]
        arr_labels = np.zeros((data_len,), dtype=np.uint8)
        if not self.arr_model:
            return arr_labels
        logger.debug("Running arrhythmia model")
        arr_len = self.arr_model.input_shape[-2]
        # Apply arrhythmia model
        for i in range(0, data_len - arr_len + 1, arr_len):
            test_x = np.expand_dims(data[i : i + arr_len], axis=(0, 1))
            y_prob = self.arr_model.predict(test_x, verbose=0)
            y_pred = 1 if y_prob[0][1] > threshold else 0
            arr_labels[i : i + arr_len] = y_pred
        # END FOR
        return arr_labels

    def segmentation_inference(
        self, data: npt.NDArray[np.float32], threshold: float = 0.75
    ) -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.uint8]]:
        """Run segmentation model on data with given threshold

        Args:
            data (npt.NDArray[np.float32]): ECG data
            threshold (float, optional): Confidence threshold. Defaults to 0.75.

        Returns:
            tuple[npt.NDArray[np.uint8], npt.NDArray[np.uint8]]: Segmentation and QRS mask
        """
        data_len = data.shape[0]
        seg_mask = np.zeros((data_len,), dtype=np.uint8)
        qrs_mask = np.zeros((data_len,), dtype=np.uint8)
        if not self.seg_model:
            return seg_mask, qrs_mask
        logger.debug("Running segmentation model")
        seg_len = self.seg_model.input_shape[-2]
        seg_olp = 20
        for i in range(0, data_len - seg_len + 1, seg_len - 2 * seg_olp):
            test_x = np.expand_dims(data[i : i + seg_len], axis=(0, 1))
            y_prob = self.seg_model.predict(test_x, verbose=0)
            y_pred = np.argmax(y_prob, axis=2)
            seg_mask[i + seg_olp : i + seg_len - seg_olp] = y_pred[0, seg_olp:-seg_olp]
            qrs_mask[i + seg_olp : i + seg_len - seg_olp] = np.where(y_prob[0, seg_olp:-seg_olp, 2] > threshold, 1, 0)
        # END FOR
        if (data_len - seg_olp) - i:
            test_x = np.expand_dims(data[-seg_len:], axis=(0, 1))
            y_prob = self.seg_model.predict(test_x, verbose=0)
            y_pred = np.argmax(y_prob, axis=2)
            seg_mask[-seg_len:-seg_olp] = y_pred[0, -seg_len:-seg_olp]
            qrs_mask[-seg_len:-seg_olp] = np.where(y_prob[0, -seg_len:-seg_olp, 2] > threshold, 1, 0)
        # END FOR
        return seg_mask, qrs_mask

    def hrv_inference(self, data: npt.NDArray[np.float32], seg_mask: npt.NDArray[np.uint8]) -> tuple:
        """Run HRV inference on data"""
        qrs_data = data.squeeze()
        qrs_data = np.where(seg_mask == HeartSegment.qrs, 10 * qrs_data, qrs_data)
        rpeaks = find_peaks(qrs_data, sample_rate=self.params.sampling_rate)
        rr_ints = compute_rr_intervals(rpeaks, sample_rate=self.params.sampling_rate)
        rr_mask = filter_rr_intervals(rr_ints, sample_rate=self.params.sampling_rate)
        return rpeaks, rr_ints, rr_mask

    def beat_inference(
        self, data: npt.NDArray[np.float32], rpeaks: npt.NDArray[np.int32], avg_rr: int, threshold: float = 0.75
    ) -> npt.NDArray[np.uint8]:
        """Run beat model on data given R-peak locations and average RR interval.

        Args:
            data (npt.NDArray[np.float32]): ECG data
            rpeaks (npt.NDArray[np.int32]): R-peak locations
            avg_rr (int): Average RR interval
            threshold (float, optional): Confidence threshold. Defaults to 0.75.
        Returns:
            npt.NDArray[np.uint8]: Beat labels
        """
        blabels = np.zeros_like(rpeaks, np.uint8)
        if not self.beat_model:
            return blabels
        logger.debug("Running beat model")
        beat_len = self.beat_model.input_shape[-2]
        for i in range(1, len(rpeaks) - 1):
            frame_start = rpeaks[i] - int(0.5 * beat_len)
            frame_end = frame_start + beat_len
            if frame_start - avg_rr < 0 or frame_end + avg_rr >= data.shape[0]:
                continue
            test_x = np.hstack(
                (
                    data[frame_start - avg_rr : frame_end - avg_rr],
                    data[frame_start:frame_end],
                    data[frame_start + avg_rr : frame_end + avg_rr],
                )
            )
            test_x = np.expand_dims(test_x, axis=(0, 1))
            y_prob = self.beat_model.predict(test_x, verbose=0).squeeze()
            y_pred = np.argmax(y_prob)
            blabels[i] = y_pred if y_prob[y_pred] > threshold else HeartBeat.noise
        # END FOR
        return blabels

    def update_app_state(self, app_state):
        """Update app state"""
        if self.hk_state.app_state == app_state:
            return
        logger.debug(f"APP_STATE={app_state}")
        self.hk_state.app_state = app_state
        try:
            self.client.set_app_state(app_state)
        except (HTTPError, ReqConnectionError, ConnectTimeout) as err:
            logger.error(f"Failed updating server {err}")

    def run(self):
        """Run inference pipeline"""
        # Grab next sample
        self.update_app_state(AppState.COLLECT_STATE)
        data = next(self.data_gen).reshape((-1, 1))

        # Pre-process
        self.update_app_state(AppState.PREPROCESS_STATE)
        data = self.preprocess(data=data)

        # Inference
        self.update_app_state(AppState.INFERENCE_STATE)

        # Apply arrhythmia model
        arr_labels = self.arrhythmia_inference(data, threshold=0.6)
        seg_mask = np.zeros((len(data),), np.uint8)
        bpm = 0
        blabels = []

        # If no arrhythmia detected, apply segmentation and HRV models
        if not np.any(arr_labels):
            # Apply segmentation model
            seg_mask, _ = self.segmentation_inference(data, threshold=0.7)

            # Apply HRV model and extract R peaks
            rpeaks, rr_ints, _ = self.hrv_inference(data, seg_mask)
            bpm = 60 / (np.mean(rr_ints) / self.params.sampling_rate)
            # bpm, _, rpeaks = compute_hrv(data, qrs_mask, self.params.sampling_rate)
            avg_rr = max(0, int(self.params.sampling_rate / (bpm / 60)))

            # Apply beat model
            blabels = self.beat_inference(data, rpeaks, avg_rr)

            # Merge beat and fiducial into single mask
            seg_mask[rpeaks] = ((blabels + 1) << 4) | seg_mask[rpeaks]
        # END IF

        self.hk_state.data_id = (self.hk_state.data_id + 1) % (2**20)
        self.hk_state.app_state = AppState.DISPLAY_STATE
        self.hk_state.data = data.squeeze().tolist()
        self.hk_state.seg_mask = seg_mask.squeeze().tolist()
        self.hk_state.results = HKResult(
            heart_rate=bpm,
            heart_rhythm=HeartRate.from_bpm(bpm).value,
            num_norm_beats=int(np.sum(blabels == HeartBeat.normal)),
            num_pac_beats=int(np.sum(blabels == HeartBeat.pac)),
            num_pvc_beats=int(np.sum(blabels == HeartBeat.pvc)),
            num_noise_beats=int(np.sum(blabels == HeartBeat.noise)),
            arrhythmia=np.any(arr_labels),
        )
        logger.debug(f"APP_STATE={self.hk_state.app_state}")
        try:
            self.client.set_state(self.hk_state)
        except (HTTPError, ConnectionError, ConnectTimeout) as err:
            logger.error(f"Failed updating server {err}")

    def startup(self):
        """Startup PC backend"""
        with console.status("[bold green] Loading models..."):
            self.load_models()
        self._run = True

    def run_forever(self):
        """Run backend"""
        self.update_app_state(AppState.IDLE_STATE)
        while self._run:
            self.run()
            time.sleep(5)

    def shutdown(self):
        """Shutdown PC backend"""
        self._run = False
