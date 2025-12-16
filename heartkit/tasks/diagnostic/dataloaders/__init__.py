import helia_edge as helia

from ....datasets import HKDataloader

from .ptbxl import PtbxlDataloader
from .lsad import LsadDataloader

DiagnosticDataloaderFactory = helia.utils.create_factory(factory="HKDiagnosticDataloaderFactory", type=HKDataloader)
DiagnosticDataloaderFactory.register("ptbxl", PtbxlDataloader)
DiagnosticDataloaderFactory.register("lsad", LsadDataloader)
