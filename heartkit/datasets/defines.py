from typing import Generator, TypeAlias

import numpy.typing as npt
import h5py

PatientGenerator = Generator[int, None, None]

PatientData: TypeAlias = dict[str, npt.NDArray] | h5py.Group
