import os
import tempfile
from enum import IntEnum, StrEnum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Extra, Field


class AugmentationParams(BaseModel, extra=Extra.allow):
    """Augmentation parameters"""

    name: str
    args: dict[str, tuple[float | int, float | int]]


class HeartTask(StrEnum):
    """Heart task"""

    arrhythmia = "arrhythmia"
    beat = "beat"
    hrv = "hrv"
    segmentation = "segmentation"


class HeartKitMode(StrEnum):
    """HeartKit Mode"""

    download = "download"
    train = "train"
    evaluate = "evaluate"
    export = "export"
    predict = "predict"
    demo = "demo"


ArchitectureType = Literal["resnet", "efficientnet", "unet", "multiresnet"]
DatasetTypes = Literal["icentia11k", "ludb", "qtdb", "synthetic", "ptbxl"]


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

    @classmethod
    def from_bpm(cls, bpm: float):
        """Assign rate based on supplied BPM."""
        if bpm < 60:
            return cls.bradycardia
        if bpm > 100:
            return cls.tachycardia
        return cls.normal


class HeartSegment(IntEnum):
    """ "Heart segment labels"""

    normal = 0
    pwave = 1
    qrs = 2
    twave = 3
    # uwave = 4  # Not used


class HeartBeatName(StrEnum):
    """Heart beat label names"""

    normal = "normal"
    pac = "pac"
    pvc = "pvc"
    noise = "noise"


class HeartRhythmName(StrEnum):
    """Heart rhythm label names"""

    normal = "normal"
    afib = "afib"
    aflut = "aflut"
    noise = "noise"


class HeartRateName(StrEnum):
    """Heart rate label names"""

    normal = "normal"
    tachycardia = "tachy"
    bradycardia = "brady"
    noise = "noise"


class HeartSegmentName(StrEnum):
    """Heart segment names"""

    normal = "normal"
    pwave = "pwave"
    qrs = "qrs"
    twave = "twave"
    uwave = "uwave"


class HeartDownloadParams(BaseModel, extra=Extra.allow):
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
    sampling_rate: int = Field(250, description="Target sampling rate (Hz)")
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
    model: str | None = Field(default=None, description="Custom model")
    model_params: dict[str, Any] | None = Field(default=None, description="Custom model parameters")

    weights_file: Path | None = Field(None, description="Path to a checkpoint weights to load")
    quantization: bool | None = Field(None, description="Enable quantization aware training (QAT)")
    # Training arguments
    batch_size: int = Field(32, description="Batch size")
    buffer_size: int = Field(100, description="Buffer size")
    epochs: int = Field(50, description="Number of epochs")
    steps_per_epoch: int | None = Field(None, description="Number of steps per epoch")
    val_metric: Literal["loss", "acc", "f1"] = Field("loss", description="Performance metric")
    augmentations: list[AugmentationParams] = Field(default_factory=list, description="Augmentations")
    # Extra arguments
    seed: int | None = Field(None, description="Random state seed")


class HeartTestParams(BaseModel, extra=Extra.allow):
    """Test command params"""

    job_dir: Path = Field(default_factory=tempfile.gettempdir, description="Job output directory")
    # Dataset arguments
    ds_path: Path = Field(default_factory=Path, description="Dataset directory")
    sampling_rate: int = Field(250, description="Target sampling rate (Hz)")
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
    sampling_rate: int = Field(250, description="Target sampling rate (Hz)")
    frame_size: int = Field(1250, description="Frame size")
    samples_per_patient: int | list[int] = Field(100, description="# test samples per patient")
    test_patients: float | None = Field(None, description="# or proportion of patients for testing")
    test_size: int = Field(100_000, description="# samples for testing")
    model_file: str | None = Field(None, description="Path to model file")
    threshold: float | None = Field(None, description="Model output threshold")
    val_acc_threshold: float | None = Field(0.98, description="Validation accuracy threshold")
    use_logits: bool = Field(True, description="Use logits output or softmax")
    quantization: bool | None = Field(None, description="Enable post training quantization (PQT)")
    tflm_var_name: str = Field("g_model", description="TFLite Micro C variable name")
    tflm_file: Path | None = Field(None, description="Path to copy TFLM header file (e.g. ./model_buffer.h)")
    data_parallelism: int = Field(
        default_factory=lambda: os.cpu_count() or 1,
        description="# of data loaders running in parallel",
    )
