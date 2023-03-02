import os
import tempfile
from enum import Enum, IntEnum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Extra, Field


class HeartTask(str, Enum):
    """Heart task"""

    rhythm = "arrhythmia"
    beat = "beat"
    hr = "hr"
    segmentation = "segmentation"


class HeartKitMode(str, Enum):
    """HeartKit Mode"""

    download = "download"
    train = "train"
    evaluate = "evaluate"
    export = "export"
    predict = "predict"
    demo = "demo"


ArchitectureType = Literal["resnet12", "resnet18", "resnet34", "resnet50", "efficientnet"]
DatasetTypes = Literal["icentia11k", "ludb", "qtdb", "synthetic"]


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
    noise = 3  # Not used


class HeartRate(IntEnum):
    """Heart rate labels"""

    normal = 0
    tachycardia = 1
    bradycardia = 2
    noise = 3  # Not used


class HeartSegment(IntEnum):
    """ "Heart segment labels"""

    normal = 0
    pwave = 1
    qrs = 2
    twave = 3
    # uwave = 4  # Not used


class HeartBeatName(str, Enum):
    """Heart beat label names"""

    normal = "normal"
    pac = "pac"
    pvc = "pvc"
    noise = "noise"


class HeartRhythmName(str, Enum):
    """Heart rhythm label names"""

    normal = "normal"
    afib = "afib"
    aflut = "aflut"
    noise = "noise"


class HeartRateName(str, Enum):
    """Heart rate label names"""

    normal = "normal"
    tachycardia = "tachy"
    bradycardia = "brady"
    noise = "noise"


class HeartSegmentName(str, Enum):
    """Heart segment names"""

    normal = "normal"
    pwave = "pwave"
    qrs = "qrs"
    twave = "twave"
    uwave = "uwave"


class HeartDownloadParams(BaseModel):
    """Download command params"""

    ds_path: Path = Field(default_factory=Path, description="Dataset root directory")
    datasets: list[DatasetTypes] = Field(default_factory=list, description="Datasets")
    progress: bool = Field(True, description="Display progress bar")
    force: bool = Field(False, description="Force download dataset- overriding existing files")
    data_parallelism: int = Field(
        default_factory=lambda: os.cpu_count() or 1,
        description="# of data loaders running in parallel",
    )


class HeartTrainParams(BaseModel, extra=Extra.allow):
    """Train command params"""

    job_dir: Path = Field(default_factory=tempfile.gettempdir, description="Job output directory")
    # Dataset arguments
    ds_path: Path = Field(default_factory=Path, description="Dataset directory")
    frame_size: int = Field(1250, description="Frame size")
    samples_per_patient: int | list[int] = Field(1000, description="# train samples per patient")
    val_samples_per_patient: int | list[int] = Field(1000, description="# validation samples per patient")
    train_patients: float | None = Field(None, description="# or proportion of patients for training")
    val_patients: float | None = Field(None, description="# or proportion of patients for validation")
    val_file: Path | None = Field(None, description="Path to load/store pickled validation file")
    val_size: int | None = Field(None, description="# samples for validation")
    data_parallelism: int = Field(
        default_factory=lambda: os.cpu_count() or 1,
        description="# of data loaders running in parallel",
    )
    # Model arguments
    weights_file: Path | None = Field(None, description="Path to a checkpoint weights to load")
    arch: ArchitectureType = Field("resnet12", description="Network architecture")
    stages: int | None = Field(None, description="# of resnet stages")
    quantization: bool | None = Field(None, description="Enable quantization aware training (QAT)")
    # Training arguments
    batch_size: int = Field(32, description="Batch size")
    buffer_size: int = Field(100, description="Buffer size")
    epochs: int = Field(50, description="Number of epochs")
    steps_per_epoch: int | None = Field(None, description="Number of steps per epoch")
    val_metric: Literal["loss", "acc", "f1"] = Field("loss", description="Performance metric")
    # Extra arguments
    seed: int | None = Field(None, description="Random state seed")


class HeartTestParams(BaseModel, extra=Extra.allow):
    """Test command params"""

    job_dir: Path = Field(default_factory=tempfile.gettempdir, description="Job output directory")
    # Dataset arguments
    ds_path: Path = Field(default_factory=Path, description="Dataset directory")
    frame_size: int = Field(1250, description="Frame size")
    samples_per_patient: int | list[int] = Field(1000, description="# test samples per patient")
    test_patients: float | None = Field(None, description="# or proportion of patients for testing")
    test_size: int = Field(200_000, description="# samples for testing")
    data_parallelism: int = Field(
        default_factory=lambda: os.cpu_count() or 1,
        description="# of data loaders running in parallel",
    )
    # Model arguments
    model_file: str | None = Field(None, description="Path to model file")
    threshold: float | None = Field(None, description="Model output threshold")
    # Extra arguments
    seed: int | None = Field(None, description="Random state seed")


class HeartExportParams(BaseModel, extra=Extra.allow):
    """Export command params"""

    job_dir: Path = Field(default_factory=tempfile.gettempdir, description="Job output directory")
    # Dataset arguments
    ds_path: Path = Field(default_factory=Path, description="Dataset directory")
    frame_size: int = Field(1250, description="Frame size")
    samples_per_patient: int | list[int] = Field(100, description="# test samples per patient")
    test_size: int = Field(100_000, description="# samples for testing")
    model_file: str | None = Field(None, description="Path to model file")
    threshold: float | None = Field(None, description="Model output threshold")
    quantization: bool | None = Field(None, description="Enable post training quantization (PQT)")
    tflm_var_name: str = Field("g_model", description="TFLite Micro C variable name")
    tflm_file: Path | None = Field(None, description="Path to copy TFLM header file (e.g. ./model_buffer.h)")
    data_parallelism: int = Field(
        default_factory=lambda: os.cpu_count() or 1,
        description="# of data loaders running in parallel",
    )


class HeartDemoParams(BaseModel, extra=Extra.allow):
    """Demo command params"""

    task: HeartTask = Field(HeartTask.rhythm, description="Heart task")
    job_dir: Path = Field(default_factory=tempfile.gettempdir, description="Job output directory")
    # Dataset arguments
    ds_path: Path = Field(default_factory=Path, description="Dataset directory")
    frame_size: int = Field(1250, description="Frame size")
    pad_size: int = Field(0, description="Pad size")
    samples_per_patient: int | list[int] = Field(1000, description="# train samples per patient")
    # EVB arguments
    vid_pid: str | None = Field(
        "51966:16385",
        description="VID and PID of serial device formatted as `VID:PID` both values in base-10",
    )
    baudrate: int = Field(115200, description="Serial baudrate")
