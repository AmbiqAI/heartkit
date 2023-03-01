from enum import Enum, IntEnum

from pydantic import BaseModel, Field


class EcgPresets(str, Enum):
    """ECG synthetic presets"""

    SR = "SR"
    ant_STEMI = "ant_STEMI"
    LAHB = "LAHB"
    LPHB = "LPHB"
    high_take_off = "high_take_off"
    LBBB = "LBBB"
    random_morphology = "random_morphology"


class SyntheticSegments(IntEnum):
    """Synthetic Segment labels"""

    background = 0
    p_wave = 1
    pr_interval = 2
    qrs_complex = 3
    st_segment = 4
    t_wave = 5
    tp_segment = 6
    tp_overlap = 7
    # Below not currently used
    p_wave_biphasic = 16
    qrs_complex_wide_rsr = 17
    qrs_complex_wide = 18
    qrs_complex_inv = 19
    qrs_complex_wide_inv_rsr = 20
    qrs_complex_wide_inv = 21
    st_segment_upsloping = 22
    st_segment_downsloping = 23
    t_wave_inv = 24


class SyntheticFiducials(IntEnum):
    """Synthetic fiducials labels"""

    p_wave = 8
    q_wave = 9
    q_trough = 10
    r_peak = 11
    rpr_peak = 12
    s_trough = 13
    j_point = 14
    qt_segment = 15  # end
    t_wave = 16  # end


class SyntheticParameters(BaseModel):
    """Synthetic ECG parameters"""
    # pylint: disable=R0902
    p_length: int = Field(80)
    pr_interval: int = Field(80)
    qrs_duration: int = Field(50)
    noisiness: float = Field(0)
    st_length: int = Field(20)
    t_length: int = Field(0)
    qt: int = Field(0)
    qtc: float = Field(0)
    flippers: list[int] = Field(default_factory=list)
    p_voltages: list[float] = Field(default_factory=list)
    p_biphasics: list[bool] = Field(default_factory=list)
    p_leans: list[float] = Field(default_factory=list)
    q_depths: list[float] = Field(default_factory=list)
    r_heights: list[float] = Field(default_factory=list)
    r_prime_presents: list[bool] = Field(default_factory=list)
    r_prime_heights: list[float] = Field(default_factory=list)
    r_to_r_prime_duration_ratio: list[float] = Field(default_factory=list)
    s_presents: list[bool] = Field(default_factory=list)
    s_depths: list[float] = Field(default_factory=list)
    s_prime_heights: list[float] = Field(default_factory=list)
    s_to_qrs_duration_ratio: list[int] = Field(default_factory=list)
    st_deltas: list[float] = Field(default_factory=list)
    j_points: list[float] = Field(default_factory=list)
    t_heights: list[float] = Field(default_factory=list)
    t_leans: list[float] = Field(default_factory=list)
