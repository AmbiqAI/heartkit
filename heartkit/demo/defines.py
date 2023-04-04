from enum import Enum

from pydantic import BaseModel, Extra, Field


class AppState(str, Enum):
    """HeartKit backend app state"""

    IDLE_STATE = "IDLE_STATE"
    COLLECT_STATE = "COLLECT_STATE"
    PREPROCESS_STATE = "PREPROCESS_STATE"
    INFERENCE_STATE = "INFERENCE_STATE"
    DISPLAY_STATE = "DISPLAY_STATE"
    FAIL_STATE = "FAIL_STATE"


class HKResult(BaseModel, extra=Extra.allow):
    """HeartKit result"""

    heart_rate: float = Field(
        default=0, description="Heart rate (BPM)", alias="heartRate"
    )
    heart_rhythm: int = Field(
        default=0, description="Heart rhythm", alias="heartRhythm"
    )
    num_norm_beats: int = Field(
        default=0, description="# normal beats", alias="numNormBeats"
    )
    num_pac_beats: int = Field(
        default=0, description="# PAC beats", alias="numPacBeats"
    )
    num_pvc_beats: int = Field(
        default=0, description="# PVC beats", alias="numPvcBeats"
    )
    arrhythmia: bool = Field(default=False, description="Arrhythmia present")


class HeartKitState(BaseModel):
    """HeartKit state"""

    data_id: int = Field(default=0, description="Data identifier", alias="dataId")
    app_state: AppState = Field(default=AppState.IDLE_STATE, alias="appState")
    data: list[float] = Field(default_factory=list, description="ECG data")
    seg_mask: list[int] = Field(
        default_factory=list, description="Segmentation mask", alias="segMask"
    )
    results: HKResult = Field(default_factory=HKResult, description="Result")
