import os
import tempfile
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field
from helia_edge.converters.tflite import QuantizationType, ConversionType


class QuantizationParams(BaseModel, extra="allow"):
    """Quantization parameters

    Attributes:
        enabled: Enable quantization
        qat: Enable quantization aware training (QAT)
        format: Quantization mode
        io_type: I/O type
        conversion: Conversion method
        debug: Debug quantization
        fallback: Fallback to float32
    """

    enabled: bool = Field(False, description="Enable quantization")
    qat: bool = Field(False, description="Enable quantization aware training (QAT)")
    format: QuantizationType = Field(QuantizationType.INT8, description="Quantization mode")
    io_type: str = Field("int8", description="I/O type")
    conversion: ConversionType = Field(ConversionType.KERAS, description="Conversion method")
    debug: bool = Field(False, description="Debug quantization")
    fallback: bool = Field(False, description="Fallback to float32")


class NamedParams(BaseModel, extra="allow"):
    """
    Named parameters is used to store parameters for a specific model, preprocessing, or augmentation.
    Typically name refers to class/method name and params is provided as kwargs.

    Attributes:
        name: Name
        params: Parameters
    """

    name: str = Field(..., description="Name")
    params: dict[str, Any] = Field(default_factory=dict, description="Parameters")


class HKMode(StrEnum):
    """heartKIT task mode"""

    download = "download"
    train = "train"
    evaluate = "evaluate"
    export = "export"
    demo = "demo"


class HKTaskParams(BaseModel, extra="allow"):
    """Task configuration params"""

    # Common arguments
    name: str = Field("experiment", description="Experiment name")
    project: str = Field("heartkit", description="Project name")
    job_dir: Path = Field(
        default_factory=lambda: Path(tempfile.gettempdir()),
        description="Job output directory",
    )

    # Dataset arguments
    datasets: list[NamedParams] = Field(default_factory=list, description="Datasets")
    dataset_weights: list[float] | None = Field(None, description="Dataset weights")
    force_download: bool = Field(False, description="Force download dataset- overriding existing files")

    # Signal arguments
    sampling_rate: int = Field(250, description="Target sampling rate (Hz)")
    frame_size: int = Field(1250, description="Frame size in samples")

    # Dataloader arguments
    samples_per_patient: int | list[int] = Field(1000, description="Number of train samples per patient")
    val_samples_per_patient: int | list[int] = Field(1000, description="Number of validation samples per patient")
    test_samples_per_patient: int | list[int] = Field(1000, description="Number of test samples per patient")

    # Preprocessing/Augmentation arguments
    preprocesses: list[NamedParams] = Field(default_factory=list, description="Preprocesses")
    augmentations: list[NamedParams] = Field(default_factory=list, description="Augmentations")

    # Class arguments
    num_classes: int = Field(1, description="Number of of classes")
    class_map: dict[int, int] = Field(default_factory=lambda: {1: 1}, description="Class/label mapping")
    class_names: list[str] | None = Field(default=None, description="Class names")

    # Split arguments
    train_patients: float | None = Field(None, description="Number of or proportion of patients for training")
    val_patients: float | None = Field(None, description="Number of or proportion of patients for validation")
    test_patients: float | None = Field(None, description="Number of or proportion of patients for testing")

    # Val/Test dataset arguments
    val_file: Path | None = Field(None, description="Path to load/store TFDS validation data")
    test_file: Path | None = Field(None, description="Path to load/store TFDS test data")
    val_size: int | None = Field(None, description="Number of samples for validation")
    test_size: int = Field(10000, description="Number of samples for testing")

    # Model arguments
    resume: bool = Field(False, description="Resume training")
    architecture: NamedParams | None = Field(default=None, description="Custom model architecture")
    model_file: Path | None = Field(None, description="Path to load/save model file (.keras)")
    use_logits: bool = Field(True, description="Use logits output or softmax")
    weights_file: Path | None = Field(None, description="Path to a checkpoint weights to load/save")
    quantization: QuantizationParams = Field(default_factory=QuantizationParams, description="Quantization parameters")

    # Training arguments
    lr_rate: float = Field(1e-3, description="Learning rate")
    lr_cycles: int = Field(3, description="Number of learning rate cycles")
    lr_decay: float = Field(0.9, description="Learning rate decay")
    label_smoothing: float = Field(0, description="Label smoothing")
    batch_size: int = Field(32, description="Batch size")
    buffer_size: int = Field(100, description="Buffer cache size")
    epochs: int = Field(50, description="Number of epochs")
    steps_per_epoch: int = Field(10, description="Number of steps per epoch")
    val_steps_per_epoch: int = Field(10, description="Number of validation steps")
    val_metric: Literal["loss", "acc", "f1"] = Field("loss", description="Performance metric")
    class_weights: list[float] | str = Field("fixed", description="Class weights")

    # Evaluation arguments
    threshold: float | None = Field(None, description="Model output threshold")
    test_metric: Literal["loss", "acc", "f1"] = Field("acc", description="Test metric")
    test_metric_threshold: float | None = Field(0.98, description="Validation metric threshold")

    # Export arguments
    tflm_var_name: str = Field("g_model", description="TFLite Micro C variable name")
    tflm_file: Path | None = Field(None, description="Path to copy TFLM header file (e.g. ./model_buffer.h)")

    # Demo arguments
    backend: str = Field("pc", description="Backend")
    demo_size: int | None = Field(1000, description="Number of samples for demo")
    display_report: bool = Field(True, description="Display report")

    # Extra arguments
    seed: int | None = Field(None, description="Random state seed")
    data_parallelism: int = Field(
        default_factory=lambda: os.cpu_count() or 1,
        description="Number of of data loaders running in parallel",
    )
    verbose: int = Field(1, ge=0, le=2, description="Verbosity level")
    model_config = ConfigDict(protected_namespaces=())

    def model_post_init(self, __context: Any) -> None:
        """Post init hook"""

        if self.val_file and len(self.val_file.parts) == 1:
            self.val_file = self.job_dir / self.val_file

        if self.test_file and len(self.test_file.parts) == 1:
            self.test_file = self.job_dir / self.test_file

        if self.model_file and len(self.model_file.parts) == 1:
            self.model_file = self.job_dir / self.model_file

        if self.weights_file and len(self.weights_file.parts) == 1:
            self.weights_file = self.job_dir / self.weights_file

        if self.tflm_file and len(self.tflm_file.parts) == 1:
            self.tflm_file = self.job_dir / self.tflm_file
