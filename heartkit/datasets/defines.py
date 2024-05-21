from typing import Callable, Generator

import numpy.typing as npt

Preprocessor = Callable[[tuple[npt.NDArray, npt.NDArray]], tuple[npt.NDArray, npt.NDArray]]

PatientGenerator = Generator[int, None, None]
