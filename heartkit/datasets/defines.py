from typing import Callable, Generator

import h5py
import numpy.typing as npt

PatientGenerator = Generator[tuple[int, h5py.Group | None], None, None]

SampleGenerator = Generator[tuple[npt.NDArray, npt.NDArray], None, None]

Preprocessor = Callable[[tuple[npt.NDArray, npt.NDArray]], tuple[npt.NDArray, npt.NDArray]]
