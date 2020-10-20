import os
from typing import Dict, List, Optional, Tuple, Type, Union
import subprocess

import numpy as np
import pandas as pd

from astropy import units as un
import lenstools
from lenstools import ConvergenceMap
import healpy as hp

from wys_ars.rays.skymap import SkyMap
from wys_ars.rays.skyio import SkyIO
from wys_ars.io import IO

class PowerSpectrum2DWarning(BaseException):
    pass


class PowerSpectrum2D:
    """
    Attributes:

    Methods:
        from_array:
        from_healpix:
        from_namaster:
        create_healpix:
    """

    def __init__(self, l: np.array, P: np.array):
        self.ell = l
        self.P = P

    @classmethod
    def from_array(
        cls,
        skymap: np.ndarray,
        multipoles: Union[List[float], np.array] = np.arange(200.0,50000.0,200.0),
    ) -> "PowerSpectrum2D":
        """
        Args:
        """
        if "kappa" in skymap.quantity:
            _map = ConvergenceMap(
                data=skymap.data[on],
                angle=skymap.opening_angle*un.deg
            )
            l, P = _map.powerSpectrum(multipoles)
        return cls(l, P, skymap, on)
    
    @classmethod
    def from_healpix(
        cls,
        skymap: SkyMap,
        of: str = "orig",
    ) -> "PowerSpectrum2D":
        """
        Get power spectrum from full-sky Healpix map.

        Args:
        """
        # measure full-sky
        Cl = hp.anafast(skymap.data[of])
        ell = np.arange(len(Cl))
        return cls(ell, Cl)
    
    @classmethod
    def from_namaster(
        cls,
        skymap: SkyMap,
        of: str = "orig",
    ) -> "PowerSpectrum2D":
        """
        Get power spectrum from spherical NaMaster map,
        but flat-sky maps also possible.

        Args:
            skymap:
            mask:
        """
        # chose multiploes for measurement
        bins = nmt.NmtBin.from_nside_linear(skymap.nside, 4)
        ell = bins.get_effective_ells()
        # measure full-sky
        Cl = nmt.compute_full_master(skymap.data[of], skymap.data[of], bins)
        return cls(ell, Cl)

    def create_healpix(
        self,
        nside: Optional[int]=None,
        rnd_seed: Optional[int]=None,
    ) -> np.ndarray:
        """
        Generate Healpix map from power spectrum.
        """
        if nside is None:
            nside = self.nside
        if nside is not None:
            np.random.seed(rnd_seed)
        skymap = hp.synfast(self.P, nside)
        return skymap

    def to_file(self, dir_out: str, extention: str = "h5") -> None:
        """ 
        Save results each power spectrum

        Args:
        """
        df = pd.DataFrame(
            data=self.P,
            index=self.l,
            columns=["P"],
        )
        filename = self._create_filename(dir_out)
        IO._remove_existing_file(filename)
        print(f"Saving results to -> {filename}")
        df.to_hdf(filename, key="df", mode="w")

    def _create_filename(self, dir_out: str) -> str:
        _filename = self.skymap._create_filename(
            self.skymap.map_file,
            self.skymap.quantity,
            on=self.on,
            extension="_",
        )
        _filename = ''.join(_filename.split("/")[-1].split(".")[:-1])
        return f"{dir_out}Cl_{_filename}.h5"
