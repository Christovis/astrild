import os
from typing import Dict, List, Optional, Tuple, Type, Union
import subprocess

import numpy as np
import pandas as pd

from astropy import units as un
import lenstools
from lenstools import ConvergenceMap

from wys_ars.rays.skymap import SkyMap
from wys_ars.rays.skyio import SkyIO
from wys_ars.io import IO

class PowerSpectrum2DWarning(BaseException):
    pass


class PowerSpectrum2D:
    """
    Attributes:

    Methods:
        compute:
    """

    def __init__(self, l: np.array, P: np.array, skymap: Type[SkyMap], on: str):
        self.l = l
        self.P = P
        self.skymap = skymap
        self.on = on

    @classmethod
    def from_array(
        cls,
        skymap: np.ndarray,
        multipoles: Union[List[float], np.array] = np.arange(200.0,50000.0,200.0),
        rtn: bool = False,
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
        skymap: np.ndarray,
        rtn: bool = False,
    ) -> "PowerSpectrum2D":
        """
        Get power spectrum from Healpix map.
        Useful for whole-sky maps.

        Args:
        """
        skymap /= (1e6 * 2.7255)            # transform [muK] to [-]
        # get healpix field properties
        nside = hp.get_nside(skymap)
        npix = hp.nside2npix(nside)
        # measure full-sky
        P = hp.anafast(skymap)
        return cls(l, P)
    
    @classmethod
    def from_namaster(
        cls,
        skymap: np.ndarray,
        maks: Optional[np.ndarray] = None,
    ) -> "PowerSpectrum2D":
        """
        Get power spectrum from NaMaster map.
        Useful for partial-sky maps.

        Args:
            skymap:
            mask:
        """
        # create NaMaster field
        skymap = nmt.NmtField(mask, [skymap])
        # chose multiploes for measurement
        b = nmt.NmtBin.from_nside_linear(nside, 4)
        ell = b.get_effective_ells()
        # measure full-sky
        P = nmt.compute_full_master(skymap, skymap, bins)
        return cls(ell, P)

    def create_healpix(self, rnd_seed: int) -> np.ndarray:
        """
        Generate Healpix map from power spectrum.
        """
        np.random.seed(rnd_seed)
        cl = np.sqrt(self.Cl)              # Cai's finding on 11/05/20
        skymap = hp.synfast(cl, nside)
        skymap = hp.pixelfunc.remove_monopole(skymap)
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
