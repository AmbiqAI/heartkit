import os
import tempfile
from enum import Enum, IntEnum
from pathlib import Path
from typing import Optional, Literal, List, Union
from pydantic import BaseModel, Field


class EcgTask(str, Enum):
    """Heart arrhythmia task"""

    rhythm = "rhythm"
    beat = "beat"
    hr = "hr"


ArchitectureType = Literal["resnet12", "resnet18", "resnet34", "resnet50"]
DatasetTypes = Literal["icentia11k"]


class HeartRhythm(IntEnum):
    """Heart rhythm labels"""

    normal = 0
    afib = 1
    aflut = 2
    noise = 3


class HeartBeat(IntEnum):
    """Heart beat labels"""

    normal = 0
    pac = 1
    aberrated = 2
    pvc = 3
    noise = 4


class HeartRate(IntEnum):
    """Heart rate labels"""

    normal = 0
    tachycardia = 1
    bradycardia = 2
    noise = 3


class HeartBeatName(str, Enum):
    """Heart beat label names"""

    normal = "normal"
    pac = "pac"
    aberrated = "aberrated"
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


class EcgDownloadParams(BaseModel):
    """Download command params"""

    db_root_path: Path = Field(
        default_factory=Path, description="Dataset root directory"
    )
    datasets: List[DatasetTypes] = Field(default_factory=list, description="Datasets")
    progress: bool = Field(True, description="Display progress bar")
    force: bool = Field(
        False, description="Force download dataset- overriding existing files"
    )
    data_parallelism: int = Field(
        default_factory=lambda: os.cpu_count() or 1,
        description="# of data loaders running in parallel",
    )


class EcgTrainParams(BaseModel):
    """Train command params"""

    # Task arguments
    task: EcgTask = Field(EcgTask.rhythm, description="Heart arrhythmia task")
    job_dir: Path = Field(
        default_factory=tempfile.gettempdir, description="Job output directory"
    )
    # Dataset arguments
    db_path: Path = Field(default_factory=Path, description="Database directory")
    frame_size: int = Field(1250, description="Frame size")
    samples_per_patient: Union[int, List[int]] = Field(
        1000, description="# train samples per patient"
    )
    val_samples_per_patient: Union[int, List[int]] = Field(
        1000, description="# validation samples per patient"
    )
    train_patients: Optional[float] = Field(
        None, description="# or proportion of patients for training"
    )
    val_patients: Optional[float] = Field(
        None, description="# or proportion of patients for validation"
    )
    val_file: Optional[Path] = Field(
        None, description="Path to load/store pickled validation file"
    )
    val_size: int = Field(200_000, description="# samples for validation")
    data_parallelism: int = Field(
        default_factory=lambda: os.cpu_count() or 1,
        description="# of data loaders running in parallel",
    )
    # Model arguments
    weights_file: Optional[Path] = Field(
        None, description="Path to a checkpoint weights to load"
    )
    arch: ArchitectureType = Field("resnet12", description="Network architecture")
    stages: Optional[int] = Field(None, description="# of resnet stages")
    quantization: Optional[bool] = Field(
        None, description="Enable quantization aware training (QAT)"
    )
    # Training arguments
    batch_size: int = Field(32, description="Batch size")
    buffer_size: int = Field(100, description="Buffer size")
    epochs: int = Field(50, description="Number of epochs")
    steps_per_epoch: int = Field(100, description="Number of steps per epoch")
    val_metric: Literal["loss", "acc", "f1"] = Field(
        "loss", description="Performance metric"
    )
    # Extra arguments
    seed: Optional[int] = Field(None, description="Random state seed")


class EcgTestParams(BaseModel):
    """Test command params"""

    # Task arguments
    task: EcgTask = Field(EcgTask.rhythm, description="Heart arrhythmia task")
    job_dir: Path = Field(
        default_factory=tempfile.gettempdir, description="Job output directory"
    )
    # Dataset arguments
    db_path: Path = Field(default_factory=Path, description="Database directory")
    frame_size: int = Field(1250, description="Frame size")
    samples_per_patient: Union[int, List[int]] = Field(
        1000, description="# test samples per patient"
    )
    test_patients: Optional[float] = Field(
        None, description="# or proportion of patients for testing"
    )
    test_size: int = Field(200_000, description="# samples for testing")
    data_parallelism: int = Field(
        default_factory=lambda: os.cpu_count() or 1,
        description="# of data loaders running in parallel",
    )
    # Model arguments
    model_file: Optional[str] = Field(None, description="Path to model file")
    threshold: Optional[float] = Field(None, description="Model output threshold")
    # Extra arguments
    seed: Optional[int] = Field(None, description="Random state seed")


class EcgDeployParams(BaseModel):
    """Deploy command params"""

    task: EcgTask = Field(EcgTask.rhythm, description="Heart arrhythmia task")
    job_dir: Path = Field(
        default_factory=tempfile.gettempdir, description="Job output directory"
    )
    # Dataset arguments
    db_path: Path = Field(default_factory=Path, description="Database directory")
    frame_size: int = Field(1250, description="Frame size")
    model_file: Optional[str] = Field(None, description="Path to model file")
    threshold: Optional[float] = Field(None, description="Model output threshold")
    quantization: Optional[bool] = Field(
        None, description="Enable post training quantization (PQT)"
    )
    tflm_var_name: str = Field("g_model", description="TFLite Micro C variable name")
    tflm_file: Optional[Path] = Field(
        None, description="Path to copy TFLM header file (e.g. ./model_buffer.h)"
    )


class EcgDemoParams(BaseModel):
    """Demo command params"""

    task: EcgTask = Field(EcgTask.rhythm, description="Heart arrhythmia task")
    job_dir: Path = Field(
        default_factory=tempfile.gettempdir, description="Job output directory"
    )
    # Dataset arguments
    db_path: Path = Field(default_factory=Path, description="Database directory")
    frame_size: int = Field(1250, description="Frame size")
    pad_size: int = Field(0, description="Frame size")
    # EVB arguments
    vid_pid: Optional[str] = Field(
        "51966:16385",
        description="VID and PID of serial device formatted as `VID:PID` both values in base-10",
    )
    baudrate: int = Field(115200, description="Serial baudrate")
