import os
import tempfile
from enum import Enum, IntEnum
from pathlib import Path
from typing import List, Literal, Optional, Union

from pydantic import BaseModel, Field


class HeartTask(str, Enum):
    """Heart task"""

    rhythm = "arrhythmia"
    beat = "beat"
    hr = "hr"
    segmentation = "segmentation"


ArchitectureType = Literal["resnet12", "resnet18", "resnet34", "resnet50"]
DatasetTypes = Literal["icentia11k", "ludb"]


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
    uwave = 4  # Not used


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


def get_class_names(task: HeartTask) -> List[str]:
    """Get class names for given task

    Args:
        task (HeartTask): Heart task

    Returns:
        List[str]: class names
    """
    if task == HeartTask.rhythm:
        # NOTE: Bucket AFIB and AFL together
        return ["NSR", "AFIB/AFL"]
    if task == HeartTask.beat:
        return ["NORMAL", "PAC", "PVC", "NOISE"]
    if task == HeartTask.hr:
        return ["NORMAL", "TACHYCARDIA", "BRADYCARDIA"]
    if task == HeartTask.segmentation:
        return ["NONE", "P-WAVE", "QRS", "T-WAVE"]
    raise ValueError(f"unknown task: {task}")


def get_num_classes(task: HeartTask) -> int:
    """Get number of classes for given task

    Args:
        task (HeartTask): Heart task

    Returns:
        int: # classes
    """
    return len(get_class_names(task=task))


class HeartDownloadParams(BaseModel):
    """Download command params"""

    ds_path: Path = Field(default_factory=Path, description="Dataset root directory")
    datasets: List[DatasetTypes] = Field(default_factory=list, description="Datasets")
    progress: bool = Field(True, description="Display progress bar")
    force: bool = Field(
        False, description="Force download dataset- overriding existing files"
    )
    data_parallelism: int = Field(
        default_factory=lambda: os.cpu_count() or 1,
        description="# of data loaders running in parallel",
    )


class HeartTrainParams(BaseModel):
    """Train command params"""

    # Task arguments
    task: HeartTask = Field(HeartTask.rhythm, description="Heart task")
    job_dir: Path = Field(
        default_factory=tempfile.gettempdir, description="Job output directory"
    )
    # Dataset arguments
    ds_path: Path = Field(default_factory=Path, description="Dataset directory")
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


class HeartTestParams(BaseModel):
    """Test command params"""

    # Task arguments
    task: HeartTask = Field(HeartTask.rhythm, description="Heart task")
    job_dir: Path = Field(
        default_factory=tempfile.gettempdir, description="Job output directory"
    )
    # Dataset arguments
    ds_path: Path = Field(default_factory=Path, description="Dataset directory")
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


class HeartExportParams(BaseModel):
    """Export command params"""

    task: HeartTask = Field(HeartTask.rhythm, description="Heart task")
    job_dir: Path = Field(
        default_factory=tempfile.gettempdir, description="Job output directory"
    )
    # Dataset arguments
    ds_path: Path = Field(default_factory=Path, description="Dataset directory")
    frame_size: int = Field(1250, description="Frame size")
    samples_per_patient: Union[int, List[int]] = Field(
        100, description="# test samples per patient"
    )
    model_file: Optional[str] = Field(None, description="Path to model file")
    threshold: Optional[float] = Field(None, description="Model output threshold")
    quantization: Optional[bool] = Field(
        None, description="Enable post training quantization (PQT)"
    )
    tflm_var_name: str = Field("g_model", description="TFLite Micro C variable name")
    tflm_file: Optional[Path] = Field(
        None, description="Path to copy TFLM header file (e.g. ./model_buffer.h)"
    )


class HeartDemoParams(BaseModel):
    """Demo command params"""

    task: HeartTask = Field(HeartTask.rhythm, description="Heart task")
    job_dir: Path = Field(
        default_factory=tempfile.gettempdir, description="Job output directory"
    )
    # Dataset arguments
    ds_path: Path = Field(default_factory=Path, description="Dataset directory")
    frame_size: int = Field(1250, description="Frame size")
    pad_size: int = Field(0, description="Pad size")
    samples_per_patient: Union[int, List[int]] = Field(
        1000, description="# train samples per patient"
    )
    # EVB arguments
    vid_pid: Optional[str] = Field(
        "51966:16385",
        description="VID and PID of serial device formatted as `VID:PID` both values in base-10",
    )
    baudrate: int = Field(115200, description="Serial baudrate")
