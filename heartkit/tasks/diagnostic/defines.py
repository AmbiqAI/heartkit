from enum import IntEnum


class HKDiagnostic(IntEnum):
    """heartKIT Diagnostic classes"""

    NORM = 0
    STTC = 1
    MI = 2
    HYP = 3
    CD = 4


# class HKDiagnostic(IntEnum):
#     """Heart rhythm labels"""

#     # NORM
#     NORM = 0  # Normal

#     # STTC
#     STTC = 100  # ST/T change
#     NST_ = 101  # Non-specific ST change
#     ISC_ = 102  # Ischemia
#     ISCA = 103  # Ischemia in anterolateral leads
#     ISCI = 104  # Ischemia in inferior leads

#     # MI
#     MI = 200  # Myocardial infarction
#     LMI = 201  # Lateral MI
#     PMI = 201  # Posterior MI
#     IMI = 202  # inferoposterolateral MI
#     AMI = 203  # anteroseptal MI

#     # HYP
#     HYP = 300  # Hypertrophy
#     RVH =  301  # Right ventricular hypertrophy
#     RAO_RAE = 302  # Right atrial overload/enlargement
#     SEHYP = 303  # Septal hypertrophy
#     AO_LAE = 304  # Left atrial overload/enlargement
#     LVH = 305 # Left ventricular hypertrophy

#     # CD
#     CD = 400  # Conductive disorder
#     LAFB_LPFB = 401  # Left anterior/posterior fascicular block
#     IRBBB = 402 # Incomplete RBBB
#     AVB = 403  # AV block
#     IVCD = 404  # Intraventricular and intra-atrial Conduction disturbances
#     CRBBB = 405  # Complete RBBB
#     CLBBB = 406  # Complete LBBB
#     WPW = 407  # Wolff-Parkinson-White syndrome
#     ILBBB = 408  #  Incomplete LBBB
