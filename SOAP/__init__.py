import warnings

warnings.filterwarnings("ignore")

import multiprocessing
import platform

from .classes import (
    CCF,
    PHOENIX,
    Spec_mu,
    ActiveRegion,
    Planet,
    Ring,
    Star,
    gaussianCCF,
    solarCCF,
    solarFTS,
    solarIAGatlas
)
from .defaults import _default_psi as psi
from .SOAP import Simulation

if __name__ == "__main__":
    os_name = platform.system()
    if os_name == "Linux":
        if multiprocessing.get_start_method(allow_none=True) != "fork":
            multiprocessing.set_start_method("fork")
    else:
        if multiprocessing.get_start_method(allow_none=True) != "spawn":
            multiprocessing.set_start_method("spawn")

__all__ = ["Simulation", "ActiveRegion", "CCF", "Star"]
