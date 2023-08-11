import os
import tempfile
from enum import IntEnum, StrEnum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Extra, Field

DatasetTypes = Literal["icentia11k", "ludb", "qtdb", "synthetic"]


class AppState(StrEnum):
    """HeartKit backend app state"""

    IDLE_STATE = "IDLE_STATE"
    COLLECT_STATE = "COLLECT_STATE"
    PREPROCESS_STATE = "PREPROCESS_STATE"
    INFERENCE_STATE = "INFERENCE_STATE"
    DISPLAY_STATE = "DISPLAY_STATE"
    FAIL_STATE = "FAIL_STATE"


class HKResult(BaseModel, extra=Extra.allow):
    """HeartKit result"""

    heart_rate: float = Field(default=0, description="Heart rate (BPM)", alias="heartRate")
    heart_rhythm: int = Field(default=0, description="Heart rhythm", alias="heartRhythm")
    num_norm_beats: int = Field(default=0, description="# normal beats", alias="numNormBeats")
    num_pac_beats: int = Field(default=0, description="# PAC beats", alias="numPacBeats")
    num_pvc_beats: int = Field(default=0, description="# PVC beats", alias="numPvcBeats")
    num_noise_beats: int = Field(default=0, description="# noise beats", alias="numNoiseBeats")
    arrhythmia: bool = Field(default=False, description="Arrhythmia present")


class HeartKitState(BaseModel):
    """HeartKit state"""

    data_id: int = Field(default=0, description="Data identifier", alias="dataId")
    app_state: AppState = Field(default=AppState.IDLE_STATE, alias="appState")
    data: list[float] = Field(default_factory=list, description="ECG data")
    seg_mask: list[int] = Field(default_factory=list, description="Segmentation mask", alias="segMask")
    results: HKResult = Field(default_factory=HKResult, description="Result")


BackendType = Literal["pc", "evb"]
FrontendType = Literal["web", "console"]


class HeartDemoParams(BaseModel, extra=Extra.allow):
    """Demo command params"""

    job_dir: Path = Field(default_factory=tempfile.gettempdir, description="Job output directory")
    rest_address: str = Field(
        default_factory=lambda: f"{os.getenv('REST_ADDR', 'http://0.0.0.0')}:{os.getenv('REST_PORT', '80')}/api/v1"
    )
    backend: BackendType = Field(default="pc", description="Backend")
    frontend: FrontendType | None = Field(default=None, description="Frontend")
    # Model paths/arguments
    arrhythmia_model: str | None = Field(default=None, description="Arrhythmia TF model path")
    segmentation_model: str | None = Field(default=None, description="Segmentation TF model path")
    beat_model: str | None = Field(default=None, description="Beat TF model path")

    # Dataset arguments
    dataset: DatasetTypes = Field(default="ludb", description="Dataset name")
    ds_path: Path = Field(default_factory=Path, description="Dataset directory")
    sampling_rate: int = Field(200, description="Target sampling rate (Hz)")
    frame_size: int = Field(2000, description="Frame size")
    pad_size: int = Field(0, description="Pad size")
    samples_per_patient: int | list[int] = Field(10, description="# train samples per patient")
    # EVB arguments
    vid_pid: str | None = Field(
        "51966:16385",
        description="VID and PID of serial device formatted as `VID:PID` in base-10",
    )
    baudrate: int = Field(115200, description="Serial baudrate")
    data_parallelism: int = Field(
        default_factory=lambda: os.cpu_count() or 1,
        description="# of data loaders running in parallel",
    )


class HeartTask(StrEnum):
    """Heart task"""

    arrhythmia = "arrhythmia"
    beat = "beat"
    hrv = "hrv"
    segmentation = "segmentation"


class HeartSegment(IntEnum):
    """ "Heart segment labels"""

    normal = 0
    pwave = 1
    qrs = 2
    twave = 3
    # uwave = 4  # Not used


class HeartRhythm(IntEnum):
    """Heart rhythm labels"""

    normal = 0
    afib = 1
    aflut = 2
    noise = 3  # Not used


class HeartBeat(IntEnum):
    """Heart beat labels"""

    normal = 0
    pac = 1
    pvc = 2
    noise = 3


def get_class_names(task: HeartTask) -> list[str]:
    """Get class names for given task

    Args:
        task (HeartTask): Heart task

    Returns:
        list[str]: class names
    """
    if task == HeartTask.arrhythmia:
        # NOTE: Bucket AFIB and AFL together
        return ["NSR", "AFIB/AFL"]
    if task == HeartTask.beat:
        return ["NORMAL", "PAC", "PVC", "NOISE"]
    if task == HeartTask.hrv:
        return ["NORMAL", "TACHYCARDIA", "BRADYCARDIA"]
    if task == HeartTask.segmentation:
        return ["NONE", "P-WAVE", "QRS", "T-WAVE"]
    raise ValueError(f"unknown task: {task}")
