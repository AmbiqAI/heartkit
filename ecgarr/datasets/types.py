from typing import Generator, Tuple
import h5py
import numpy.typing as npt

PatientGenerator = Generator[Tuple[int, h5py.Group], None, None]

SampleGenerator = Generator[Tuple[npt.ArrayLike, h5py.Group], None, None]
