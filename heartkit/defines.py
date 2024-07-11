import os
import tempfile
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field
from keras_edge.converters.tflite import QuantizationType


class QuantizationParams(BaseModel, extra="allow"):
    """Quantization parameters"""

    enabled: bool = Field(False, description="Enable quantization")
    qat: bool = Field(False, description="Enable quantization aware training (QAT)")
    mode: QuantizationType = Field(QuantizationType.INT8, description="Quantization mode")
    io_type: str = Field("int8", description="I/O type")
    concrete: bool = Field(True, description="Use concrete function")
    debug: bool = Field(False, description="Debug quantization")
    fallback: bool = Field(False, description="Fallback to float32")


class ModelArchitecture(BaseModel, extra="allow"):
    """Model architecture parameters"""

    name: str
    params: dict[str, Any] = Field(default_factory=dict, description="Parameters")


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
    path: Path = Field(default_factory=Path, description="Dataset path")
    params: dict[str, Any] = Field(default_factory=dict, description="Parameters")
    weight: float = Field(1, description="Dataset weight")


class HKMode(StrEnum):
    """HeartKit Mode"""

    download = "download"
    train = "train"
    evaluate = "evaluate"
    export = "export"
    demo = "demo"


class HKDownloadParams(BaseModel, extra="allow"):
    """Download command params"""

    job_dir: Path = Field(
        default_factory=lambda: Path(tempfile.gettempdir()),
        description="Job output directory",
    )
    datasets: list[DatasetParams] = Field(default_factory=list, description="Datasets")
    progress: bool = Field(True, description="Display progress bar")
    force: bool = Field(False, description="Force download dataset- overriding existing files")
    data_parallelism: int = Field(
        default_factory=lambda: os.cpu_count() or 1,
        description="# of data loaders running in parallel",
    )


class HKTrainParams(BaseModel, extra="allow"):
    """Train command params"""

    name: str = Field("experiment", description="Experiment name")
    project: str = Field("heartkit", description="Project name")
    job_dir: Path = Field(
        default_factory=lambda: Path(tempfile.gettempdir()),
        description="Job output directory",
    )
    # Dataset arguments
    datasets: list[DatasetParams] = Field(default_factory=list, description="Datasets")

    sampling_rate: int = Field(250, description="Target sampling rate (Hz)")
    frame_size: int = Field(1250, description="Frame size")
    num_classes: int = Field(1, description="# of classes")
    class_map: dict[int, int] = Field(default_factory=lambda: {1: 1}, description="Class/label mapping")
    class_names: list[str] | None = Field(default=None, description="Class names")

    samples_per_patient: int | list[int] = Field(1000, description="# train samples per patient")
    val_samples_per_patient: int | list[int] = Field(1000, description="# validation samples per patient")
    train_patients: float | None = Field(None, description="# or proportion of patients for training")
    val_patients: float | None = Field(None, description="# or proportion of patients for validation")
    val_file: Path | None = Field(None, description="Path to load/store pickled validation file")
    val_size: int | None = Field(None, description="# samples for validation")

    # Model arguments
    resume: bool = Field(False, description="Resume training")
    architecture: ModelArchitecture | None = Field(default=None, description="Custom model architecture")
    model_file: Path | None = Field(None, description="Path to save model file (.keras)")
    threshold: float | None = Field(None, description="Model output threshold")

    weights_file: Path | None = Field(None, description="Path to a checkpoint weights to load")
    quantization: QuantizationParams = Field(default_factory=QuantizationParams, description="Quantization parameters")
    # Training arguments
    lr_rate: float = Field(1e-3, description="Learning rate")
    lr_cycles: int = Field(3, description="Number of learning rate cycles")
    lr_decay: float = Field(0.9, description="Learning rate decay")
    class_weights: Literal["balanced", "fixed"] = Field("fixed", description="Class weights")
    label_smoothing: float = Field(0, description="Label smoothing")
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

    def model_post_init(self, __context: Any) -> None:
        """Post init hook"""

        if self.val_file and len(self.val_file.parts) == 1:
            self.val_file = self.job_dir / self.val_file

        if self.model_file and len(self.model_file.parts) == 1:
            self.model_file = self.job_dir / self.model_file

        if self.weights_file and len(self.weights_file.parts) == 1:
            self.weights_file = self.job_dir / self.weights_file


class HKTestParams(BaseModel, extra="allow"):
    """Test command params"""

    name: str = Field("experiment", description="Experiment name")
    project: str = Field("heartkit", description="Project name")
    job_dir: Path = Field(
        default_factory=lambda: Path(tempfile.gettempdir()),
        description="Job output directory",
    )
    # Dataset arguments
    datasets: list[DatasetParams] = Field(default_factory=list, description="Datasets")
    sampling_rate: int = Field(250, description="Target sampling rate (Hz)")
    frame_size: int = Field(1250, description="Frame size")
    num_classes: int = Field(1, description="# of classes")
    class_map: dict[int, int] = Field(default_factory=lambda: {1: 1}, description="Class/label mapping")
    class_names: list[str] | None = Field(default=None, description="Class names")
    test_samples_per_patient: int | list[int] = Field(1000, description="# test samples per patient")
    test_patients: float | None = Field(None, description="# or proportion of patients for testing")
    test_size: int = Field(200_000, description="# samples for testing")
    test_file: Path | None = Field(None, description="Path to load/store pickled test file")
    preprocesses: list[PreprocessParams] = Field(default_factory=list, description="Preprocesses")
    augmentations: list[AugmentationParams] = Field(default_factory=list, description="Augmentations")
    # Model arguments
    model_file: Path | None = Field(None, description="Path to save model file (.keras)")
    threshold: float | None = Field(None, description="Model output threshold")
    # Extra arguments
    seed: int | None = Field(None, description="Random state seed")
    data_parallelism: int = Field(
        default_factory=lambda: os.cpu_count() or 1,
        description="# of data loaders running in parallel",
    )
    model_config = ConfigDict(protected_namespaces=())

    def model_post_init(self, __context: Any) -> None:
        """Post init hook"""

        if self.test_file and len(self.test_file.parts) == 1:
            self.test_file = self.job_dir / self.test_file

        if self.model_file and len(self.model_file.parts) == 1:
            self.model_file = self.job_dir / self.model_file


class HKExportParams(BaseModel, extra="allow"):
    """Export command params"""

    name: str = Field("experiment", description="Experiment name")
    project: str = Field("heartkit", description="Project name")
    job_dir: Path = Field(
        default_factory=lambda: Path(tempfile.gettempdir()),
        description="Job output directory",
    )
    # Dataset arguments
    datasets: list[DatasetParams] = Field(default_factory=list, description="Datasets")
    sampling_rate: int = Field(250, description="Target sampling rate (Hz)")
    frame_size: int = Field(1250, description="Frame size")
    num_classes: int = Field(3, description="# of classes")
    class_map: dict[int, int] = Field(default_factory=lambda: {1: 1}, description="Class/label mapping")
    class_names: list[str] | None = Field(default=None, description="Class names")
    test_samples_per_patient: int | list[int] = Field(100, description="# test samples per patient")
    test_patients: float | None = Field(None, description="# or proportion of patients for testing")
    test_size: int = Field(100_000, description="# samples for testing")
    test_file: Path | None = Field(None, description="Path to load/store pickled test file")
    preprocesses: list[PreprocessParams] = Field(default_factory=list, description="Preprocesses")
    augmentations: list[AugmentationParams] = Field(default_factory=list, description="Augmentations")
    model_file: Path | None = Field(None, description="Path to save model file (.keras)")
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

    def model_post_init(self, __context: Any) -> None:
        """Post init hook"""

        if self.test_file and len(self.test_file.parts) == 1:
            self.test_file = self.job_dir / self.test_file

        if self.model_file and len(self.model_file.parts) == 1:
            self.model_file = self.job_dir / self.model_file

        if self.tflm_file and len(self.tflm_file.parts) == 1:
            self.tflm_file = self.job_dir / self.tflm_file


class HKDemoParams(BaseModel, extra="allow"):
    """HK demo command params"""

    name: str = Field("experiment", description="Experiment name")
    project: str = Field("heartkit", description="Project name")
    job_dir: Path = Field(
        default_factory=lambda: Path(tempfile.gettempdir()),
        description="Job output directory",
    )
    # Dataset arguments
    datasets: list[DatasetParams] = Field(default_factory=list, description="Datasets")
    sampling_rate: int = Field(250, description="Target sampling rate (Hz)")
    frame_size: int = Field(1250, description="Frame size")
    num_classes: int = Field(1, description="# of classes")
    class_map: dict[int, int] = Field(default_factory=lambda: {1: 1}, description="Class/label mapping")
    class_names: list[str] | None = Field(default=None, description="Class names")
    preprocesses: list[PreprocessParams] = Field(default_factory=list, description="Preprocesses")
    augmentations: list[AugmentationParams] = Field(default_factory=list, description="Augmentations")
    # Model arguments
    model_file: Path | None = Field(None, description="Path to save model file (.keras)")
    backend: str = Field("pc", description="Backend")
    # Demo arguments
    demo_size: int | None = Field(1000, description="# samples for demo")
    display_report: bool = Field(True, description="Display report")
    # Extra arguments
    seed: int | None = Field(None, description="Random state seed")
    model_config = ConfigDict(protected_namespaces=())

    def model_post_init(self, __context: Any) -> None:
        """Post init hook"""

        if self.model_file and len(self.model_file.parts) == 1:
            self.model_file = self.job_dir / self.model_file
