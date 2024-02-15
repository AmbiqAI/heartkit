import os
import tempfile
from enum import IntEnum, StrEnum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class QuantizationParams(BaseModel, extra="allow"):
    """Quantization parameters"""

    enabled: bool = Field(False, description="Enable quantization")
    qat: bool = Field(False, description="Enable quantization aware training (QAT)")
    ptq: bool = Field(False, description="Enable post training quantization (PTQ)")
    input_type: str | None = Field(None, description="Input type")
    output_type: str | None = Field(None, description="Output type")
    supported_ops: list[str] | None = Field(None, description="Supported ops")


class ModelArchitecture(BaseModel, extra="allow"):
    """Model architecture parameters"""

    name: str
    params: dict[str, Any]


class PreprocessParams(BaseModel, extra="allow"):
    """Preprocessing parameters"""

    name: str
    params: dict[str, Any]


class AugmentationParams(BaseModel, extra="allow"):
    """Augmentation parameters"""

    name: str
    params: dict[str, tuple[float | int, float | int]]


class DatasetParams(BaseModel, extra="allow"):
    """Dataset parameters"""

    name: str
    params: dict[str, Any]


class HKMode(StrEnum):
    """HeartKit Mode"""

    download = "download"
    train = "train"
    evaluate = "evaluate"
    export = "export"
    demo = "demo"


class HeartSegment(IntEnum):
    """Heart segment labels"""

    normal = 0
    pwave = 1
    qrs = 2
    twave = 3
    uwave = 4  # Not used


class HeartSegmentName(StrEnum):
    """Heart segment names"""

    normal = "normal"
    pwave = "pwave"
    qrs = "qrs"
    twave = "twave"
    uwave = "uwave"  # Not used


class HeartRhythm(IntEnum):
    """Heart rhythm labels"""

    normal = 0
    afib = 1
    aflut = 2
    noise = 3  # Not used


class HeartRhythmName(StrEnum):
    """Heart rhythm label names"""

    normal = "normal"
    afib = "afib"
    aflut = "aflut"
    noise = "noise"


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


class HeartRateName(StrEnum):
    """Heart rate label names"""

    normal = "normal"
    tachycardia = "tachy"
    bradycardia = "brady"
    noise = "noise"


class HeartBeat(IntEnum):
    """Heart beat labels"""

    normal = 0
    pac = 1
    pvc = 2
    noise = 3  # Not used


class HeartBeatName(StrEnum):
    """Heart beat label names"""

    normal = "normal"
    pac = "pac"
    pvc = "pvc"
    noise = "noise"


class HKDownloadParams(BaseModel, extra="allow"):
    """Download command params"""

    job_dir: Path = Field(default_factory=tempfile.gettempdir, description="Job output directory")
    ds_path: Path = Field(default_factory=Path, description="Dataset root directory")
    datasets: list[str] = Field(default_factory=list, description="Datasets")
    progress: bool = Field(True, description="Display progress bar")
    force: bool = Field(False, description="Force download dataset- overriding existing files")
    data_parallelism: int = Field(
        default_factory=lambda: os.cpu_count() or 1,
        description="# of data loaders running in parallel",
    )


class HKTrainParams(BaseModel, extra="allow"):
    """Train command params"""

    job_dir: Path = Field(default_factory=tempfile.gettempdir, description="Job output directory")
    # Dataset arguments
    ds_path: Path = Field(default_factory=Path, description="Dataset directory")
    datasets: list[DatasetParams] = Field(default_factory=list, description="Datasets")
    sampling_rate: int = Field(250, description="Target sampling rate (Hz)")
    frame_size: int = Field(1250, description="Frame size")
    num_classes: int = Field(3, description="# of classes")
    samples_per_patient: int | list[int] = Field(1000, description="# train samples per patient")
    val_samples_per_patient: int | list[int] = Field(1000, description="# validation samples per patient")
    train_patients: float | None = Field(None, description="# or proportion of patients for training")
    val_patients: float | None = Field(None, description="# or proportion of patients for validation")
    val_file: Path | None = Field(None, description="Path to load/store pickled validation file")
    val_size: int | None = Field(None, description="# samples for validation")
    # Model arguments
    resume: bool = Field(False, description="Resume training")
    architecture: ModelArchitecture | None = Field(default=None, description="Custom model architecture")
    model_file: str | None = Field(None, description="Path to save model file (.keras)")

    weights_file: Path | None = Field(None, description="Path to a checkpoint weights to load")
    quantization: QuantizationParams = Field(default_factory=QuantizationParams, description="Quantization parameters")
    # Training arguments
    lr_rate: float = Field(1e-3, description="Learning rate")
    lr_cycles: int = Field(3, description="Number of learning rate cycles")
    lr_decay: float = Field(0.9, description="Learning rate decay")
    class_weights: Literal["balanced", "fixed"] = Field("fixed", description="Class weights")
    batch_size: int = Field(32, description="Batch size")
    buffer_size: int = Field(100, description="Buffer size")
    epochs: int = Field(50, description="Number of epochs")
    steps_per_epoch: int = Field(10, description="Number of steps per epoch")
    val_metric: Literal["loss", "acc", "f1"] = Field("loss", description="Performance metric")
    # Preprocessing/Augmentation arguments
    preprocesses: list[PreprocessParams] = Field(default_factory=list, description="Preprocesses")
    augmentations: list[AugmentationParams] = Field(default_factory=list, description="Augmentations")
    # Extra arguments
    seed: int | None = Field(None, description="Random state seed")
    data_parallelism: int = Field(
        default_factory=lambda: os.cpu_count() or 1,
        description="# of data loaders running in parallel",
    )
    model_config = ConfigDict(protected_namespaces=())


class HKTestParams(BaseModel, extra="allow"):
    """Test command params"""

    job_dir: Path = Field(default_factory=tempfile.gettempdir, description="Job output directory")
    # Dataset arguments
    ds_path: Path = Field(default_factory=Path, description="Dataset directory")
    datasets: list[DatasetParams] = Field(default_factory=list, description="Datasets")
    sampling_rate: int = Field(250, description="Target sampling rate (Hz)")
    frame_size: int = Field(1250, description="Frame size")
    num_classes: int = Field(3, description="# of classes")
    test_samples_per_patient: int | list[int] = Field(1000, description="# test samples per patient")
    test_patients: float | None = Field(None, description="# or proportion of patients for testing")
    test_size: int = Field(200_000, description="# samples for testing")
    preprocesses: list[PreprocessParams] = Field(default_factory=list, description="Preprocesses")
    augmentations: list[AugmentationParams] = Field(default_factory=list, description="Augmentations")
    # Model arguments
    model_file: str | None = Field(None, description="Path to model file")
    threshold: float | None = Field(None, description="Model output threshold")
    # Extra arguments
    seed: int | None = Field(None, description="Random state seed")
    data_parallelism: int = Field(
        default_factory=lambda: os.cpu_count() or 1,
        description="# of data loaders running in parallel",
    )
    model_config = ConfigDict(protected_namespaces=())


class HKExportParams(BaseModel, extra="allow"):
    """Export command params"""

    job_dir: Path = Field(default_factory=tempfile.gettempdir, description="Job output directory")
    # Dataset arguments
    ds_path: Path = Field(default_factory=Path, description="Dataset directory")
    datasets: list[DatasetParams] = Field(default_factory=list, description="Datasets")
    sampling_rate: int = Field(250, description="Target sampling rate (Hz)")
    frame_size: int = Field(1250, description="Frame size")
    num_classes: int = Field(3, description="# of classes")
    preprocesses: list[PreprocessParams] = Field(default_factory=list, description="Preprocesses")
    augmentations: list[AugmentationParams] = Field(default_factory=list, description="Augmentations")
    test_samples_per_patient: int | list[int] = Field(100, description="# test samples per patient")
    test_patients: float | None = Field(None, description="# or proportion of patients for testing")
    test_size: int = Field(100_000, description="# samples for testing")
    model_file: str | None = Field(None, description="Path to model file")
    threshold: float | None = Field(None, description="Model output threshold")
    val_acc_threshold: float | None = Field(0.98, description="Validation accuracy threshold")
    use_logits: bool = Field(True, description="Use logits output or softmax")
    quantization: QuantizationParams = Field(default_factory=QuantizationParams, description="Quantization parameters")
    tflm_var_name: str = Field("g_model", description="TFLite Micro C variable name")
    tflm_file: Path | None = Field(None, description="Path to copy TFLM header file (e.g. ./model_buffer.h)")
    data_parallelism: int = Field(
        default_factory=lambda: os.cpu_count() or 1,
        description="# of data loaders running in parallel",
    )
    model_config = ConfigDict(protected_namespaces=())


class HKDemoParams(BaseModel, extra="allow"):
    """HK demo command params"""

    job_dir: Path = Field(default_factory=tempfile.gettempdir, description="Job output directory")
    # Dataset arguments
    ds_path: Path = Field(default_factory=Path, description="Dataset base directory")
    datasets: list[DatasetParams] = Field(default_factory=list, description="Datasets")
    sampling_rate: int = Field(250, description="Target sampling rate (Hz)")
    frame_size: int = Field(1250, description="Frame size")
    num_classes: int = Field(3, description="# of classes")
    preprocesses: list[PreprocessParams] = Field(default_factory=list, description="Preprocesses")
    augmentations: list[AugmentationParams] = Field(default_factory=list, description="Augmentations")
    # Model arguments
    model_file: str | None = Field(None, description="Path to model file (.keras, .h5, or .tflite)")
    backend: Literal["pc", "evb"] = Field("pc", description="Backend")
    # Extra arguments
    seed: int | None = Field(None, description="Random state seed")
    model_config = ConfigDict(protected_namespaces=())
