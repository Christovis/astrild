import os
from typing import Dict, List, Optional, Tuple, Type, Union
import subprocess

import numpy as np
import pandas as pd

from astropy import units as un
import lenstools
from lenstools import ConvergenceMap

from wys_ars import io as IO
from wys_ars.rays.skymap import SkyMap

class Bispectrum2DWarning(BaseException):
    pass


class Bispectrum2D:
    """
    Attributes:

    Methods:
        compute:
    """

    def __init__(self, l: np.array, B: np.array, skymap: Type[SkyMap], on: str):
        self.l = l
        self.B = B
        self.skymap = skymap
        self.on = on

    @classmethod
    def from_skymap(
        cls,
        skymap: Type[SkyMap],
        on: str,
        multipoles: Union[List[float], np.array] = np.arange(200.0,50000.0,200.0),
        rtn: bool = False,
    ) -> "Bispectrum2D":
        """
        Args:
        """
        if "kappa" in skymap.quantity:
            _map = ConvergenceMap(
                data=skymap.data[on],
                angle=skymap.opening_angle*un.deg
            )
            l, B = _map.bispectrum(multipoles, configuration="equilateral")
        return cls(l, B, skymap, on)

    def to_file(self, dir_out: str, extention: str = "h5") -> None:
        """ 
        Save results each bispectrum

        Args:
        """
        df = pd.DataFrame(
            data=self.B,
            index=self.l,
            columns=["B"],
        )
        filename = self._create_filename(dir_out)
        IO._remove_existing_file(filename)
        print(f"Saving results to -> {filename}")
        df.to_hdf(filename, key="df", mode="w")

    def _create_filename(self, dir_out: str) -> str:
        _filename = self.skymap._create_filename(
            self.skymap.map_file,
            self.skymap.quantity,
            self.on,
            extension="_",
        )
        _filename = ''.join(_filename.split("/")[-1].split(".")[:-1])
        return f"{dir_out}Bl_{_filename}.h5"
