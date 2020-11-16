import os
from typing import Dict, List, Optional, Tuple, Type, Union
import subprocess

import numpy as np
import pandas as pd

import healpy as hp
from astropy import units as un
import lenstools
from lenstools import ConvergenceMap

from wys_ars.rays.skys import SkyNamaster
from wys_ars.rays.skys import SkyHealpix
from wys_ars.rays.skys import SkyArray
from wys_ars.rays.skyio import SkyIO
from wys_ars.io import IO

class PowerSpectrum2DWarning(BaseException):
    pass


class AngularPowerSpectrum:
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
        skymap: Type[SkyArray],
        on: str,
        multipoles: Union[List[float], np.array] = np.arange(200.0,50000.0,200.0),
    ) -> "PowerSpectrum2D":
        """
        Args:
        """
        _map = ConvergenceMap(
            data=skymap.data[on],
            angle=skymap.opening_angle*un.deg
        )
        l, P = _map.powerSpectrum(multipoles)
        return cls(l, P)
    
    @classmethod
    def from_healpix(
        cls,
        skymap: np.ndarray,
        ell_lim: Optional[tuple] = None,
    ) -> "AngularPowerSpectrum":
        """
        Get power spectrum from full-sky Healpix map.

        Args:
        """
        # measure full-sky
        Cl = hp.anafast(skymap)
        ell = np.arange(len(Cl), dtype=float)
        if ell_lim:
            idx = np.logical_and(np.min(ell_lim) < ell, ell < np.max(ell_lim))
            ell = ell[idx]
            Cl = Cl[idx]
        return cls(ell, Cl)
    
    @classmethod
    def from_namaster(
        cls,
        skyfield: np.ndarray,
        nside: Optional[int] = None,
    ) -> "AngularPowerSpectrum":
        """
        Get power spectrum from spherical NaMaster map,
        but flat-sky maps also possible.

        Args:
            skymap:
            mask:
        """
        # chose multiploes for measurement
        nside = hp.npix2nside(np.arange(len(skymap)))
        bins = nmt.NmtBin.from_nside_linear(nside, 4)
        ell = bins.get_effective_ells()
        # measure full-sky
        Cl = nmt.compute_full_master(skyfield, skyfield, bins)
        return cls(ell, Cl)
    
    def to_skyhealpix(
        self,
        nside: Optional[int]=None,
        lmax: Optional[int]=None,
        rnd_seed: Optional[int]=None,
    ) -> Type[SkyHealpix]:
        """
        Generate Healpix map from power spectrum.
        """
        if nside is None:
            nside = self.nside
        if nside is not None:
            np.random.seed(rnd_seed)

        return SkyHealpix.from_Cl_array(self.P, quantity=None, dir_in=None,)

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
