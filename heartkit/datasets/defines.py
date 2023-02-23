from typing import Generator

import h5py
import numpy.typing as npt

PatientGenerator = Generator[tuple[int, h5py.Group], None, None]

SampleGenerator = Generator[tuple[npt.ArrayLike, h5py.Group], None, None]
