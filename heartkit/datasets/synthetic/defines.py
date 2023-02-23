from enum import IntEnum

from pydantic import BaseModel, Field

# ecg_presets: list[str] = [
#     "LAHB",
#     "LPHB",
#     "high_take_off",
#     "LBBB",
#     "ant_STEMI",
#     "random_morphology",
# ]

# segmentation_key: dict[int, str] = {
#     0: "background",
#     1: "P wave",
#     # 16 : "Biphasic P wave",
#     2: "PR interval",
#     3: "QRS complex",
#     # 17 : "Broad QRS with RSR pattern",
#     # 18 : "Broad QRS without RSR pattern",
#     # 19 : "Inverted narrow QRS complex",
#     # 20 : "Inverted QRS with RSR pattern",
#     # 21 : "Inverted QRS without RSR pattern",
#     4: "ST segment",
#     # 22 : "Upsloping ST segment",
#     # 23 : "Downsloping ST segment",
#     5: "T wave",
#     # 24 : "Inverted T wave",
#     6: "T-P segment",
#     7: "T/P overlap",
#     8: "start of P",
#     9: "start of Q wave",
#     10: "Q trough",
#     11: "R peak",
#     12: "R prime peak",
#     13: "S trough",
#     14: "J point",
#     15: "end of QT segment (max slope method)",
#     16: "end of T wave (absolute method)",
# }

# # the classes that represent ECG points rather than waves:
# # (note that point 25 - T wave end - is included in waves as it is used to
# # create bounding boxes)
# points: list[int] = [8, 9, 10, 11, 12, 13, 14, 15]
# # points that should be present for every beat:
# vital_points: list[int] = [9, 11, 14]
# # maps points to waves:
# point_mapper: dict[int, str] = {
#     8: "P wave",
#     9: "QRS",
#     10: "QRS",
#     11: "QRS",
#     12: "QRS",
#     13: "QRS",
#     14: "QRS",
#     15: 5,
# }
# # the classes that represent ECG waves:
# waves = np.setdiff1d([*range(len(segmentation_key))], points).tolist()
# n_classes = len(waves) - 1

Preset = Literal[
    "SR", "ant_STEMI", "LAHB", "LPHB", "high_take_off", "LBBB", "random_morphology"
]


class SyntheticSegments(IntEnum):
    background = 0
    p_wave = 1
    pr_interval = 2
    qrs_complex = 3
    st_segment = 4
    t_wave = 5
    tp_segment = 6
    tp_overlap = 7
    #
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
