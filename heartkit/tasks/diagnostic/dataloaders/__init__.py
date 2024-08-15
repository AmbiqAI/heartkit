import neuralspot_edge as nse

from ....datasets import HKDataloader

from .ptbxl import PtbxlDataloader
from .lsad import LsadDataloader

DiagnosticDataloaderFactory = nse.utils.create_factory(factory="HKDiagnosticDataloaderFactory", type=HKDataloader)
DiagnosticDataloaderFactory.register("ptbxl", PtbxlDataloader)
DiagnosticDataloaderFactory.register("lsad", LsadDataloader)
