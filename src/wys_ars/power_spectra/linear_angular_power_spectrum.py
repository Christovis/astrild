import os
from typing import Dict, List, Optional, Tuple, Type, Union
import subprocess

import numpy as np
import numba as nb
from scipy.integrate import quad
import pandas as pd

from joblib import Parallel, delayed
import multiprocessing

import healpy as hp
from astropy import units as un
from astropy import constants as const
from astropy.cosmology import LambdaCDM, z_at_value
import camb
import lenstools
from lenstools import ConvergenceMap

from wys_ars.power_spectra.linear_power_spectrum import LinearPowerSpectrum
from wys_ars.io import IO

# store available nr. of cpus for parallel computation
ncpus_available = multiprocessing.cpu_count()

class LinearAngularPowerSpectrumWarning(BaseException):
    pass


class LinearAngularPowerSpectrum:
    """
    Attributes:
        z_range:
        ell_range:
        cambcosmo:
        astropycosmo:
        lps:

    Methods:
        create_healpix:
    """

    def __init__(
        self,
        z_range: np.array,
        ell_range: np.array,
        cambcosmo: camb.CAMBparams,
        astropycosmo: LambdaCDM,
        lps: LinearPowerSpectrum,
        ncpus: int,
    ):
        self._z_range = z_range
        self._ell_range = ell_range
        self.cambcosmo = cambcosmo
        self.astropycosmo = astropycosmo
        self.lps = lps
        if ncpus == 0:
            self._ncpus = ncpus_available
        else:
            self._ncpus = ncpus
        self._C_tt_outdated = True

    @classmethod
    def from_parameters(
        cls,
        z_array: np.array,
        ell_array: np.array,
        H0: float = 67.74,
        Om0: float = 0.3089,
        Ob0: float = 0.0482754208891869,
        Ode0: float = 0.6911,
        T0_cmb: float = 2.7255,
        As: float = 2.105e-9,
        ns: float = 0.9665,
        ncpus: int = 1,
    ) -> "LinearAngularPowerSpectrum":
        """
        Args:
            H0:
                Hubble constant today.
            Om0:
                Matter density today.
            Ob0:
                Baryon density today.
            Ode0:
                Dark Energy density today.
            T0_cmb:
                CMB temperature today.
            As:
            ns:
        
        Returns:
        """
        dic = {
            "H0": H0,
            "Om0": Om0,
            "Ob0": Ob0,
            "Ode0": Ode0,
            "T0_cmb": T0_cmb,
            "As": As,
            "ns": ns,
        }
        return cls.from_dictionary(z_array, ell_array, dic, ncpus)
    
    @classmethod
    def from_dictionary(
        cls,
        z_array: np.array,
        ell_array: np.array,
        dic: dict,
        ncpus: int,
    ) -> "LinearAngularPowerSpectrum":
        # initialize linear power spetrum class
        cambcosmo = cls.init_camb_cosmos_from_astropy_cosmos(dic)
        lps = LinearPowerSpectrum.from_dictionary(dic)
        astropycosmo = LambdaCDM(**dic)
        return cls(z_array, ell_array, cambcosmo, astropycosmo, lps, ncpus)

    @staticmethod
    def init_camb_cosmos_from_astropy_cosmos(dic: dict,) -> camb.CAMBparams:
        cambcosmo = camb.CAMBparams()
        cambcosmo.set_cosmology(
            H0=dic["H0"],
            ombh2=dic["Ob0"] * (dic["H0"]/100)**2,
            omch2=(dic["Om0"]-dic["Ob0"]) * (dic["H0"]/100)**2,
            TCMB=dic["T0_cmb"],
        )
        cambcosmo.InitPower.set_params(As=dic["As"], ns=dic["As"])   
        return cambcosmo

    @property
    def z_range(self):
        return self._z_range
 
    @z_range.setter
    def z_range(self, val: np.array):
        self._z_range = val
        self._cl_outdated = True
 
    @property
    def ell(self):
        return self._ell_range
    
    @ell.setter
    def ell(self, val: np.array):
        self._ell_range = val
        self._C_tt_outdated = True
    
    @property
    def ncpus(self):
        return self._ncpus
    
    @ncpus.setter
    def ncpus(self, val: int):
        if (ncpus == 0) or (ncpus < -1):
            raise ValueError(
                f"ncpus={ncpus} is not valid. Please enter a value " +\
                ">0 for ncpus or -1 to use all available cores."
            )
        elif ncpus == -1:
            self._ncpus = ncpus_available
        else:
            self._ncpus = val
        
    @property
    def Cl(self):
        if self._C_tt_outdated:
            self.compute_C_tt()
        return self._C_tt
    
    
    def compute_C_tt(self) -> None:
        """
        Angular auto temperature fluctuation power spectrum.

        Args:
            ncpus:
                Nr. of CPUs. If ncpus=0 than it will use all existing CPUs.
        """

        def integration(ell):
            return quad(
                self.p_dpdp_integrant,
                self._z_range.min(),
                self._z_range.max(),
                args=(
                    ell,
                    self.astropycosmo,
                    self.lps,
                ),
                epsabs=0, epsrel=1e-6
            )[0]

        print(f"just about to start with {self._ncpus}")
        _cl_tt = Parallel(n_jobs=self._ncpus,)(
            delayed(integration)(ell) for ell in self._ell_range
        )

        self._C_tt = np.array(_cl_tt) * 4/const.c.to(un.km/un.second).value**5
        self._C_tt_outdated = False

    @staticmethod
    def p_dpdp_integrant(
        dx: float,
        ell: float,
        astropycosmo: LambdaCDM,
        lps: LinearPowerSpectrum,
    ) -> np.array:
        """
        Function of linear density P(k) to be integrated over time
        Eq. 2 in arxiv:0809.4488

        Args:
        Returns:
        """
        z = dx
        prefac = 1/(1 + z)**2

        chi = astropycosmo.comoving_distance(z).to(
            un.Mpc/un.littleh, un.with_H0(astropycosmo.H0)
        ).value  #[Mpc/h] comov. dist.
        k = ell/chi  #[h/Mpc] wavenumber (Limber-approx.)
        # power spectrum of linear ISW effect (time derivative of potential)
        return prefac/(chi**2) * lps.P_dpdp(z, k)
   
    def compute_C_kk(
        self, redshift: np.array, ells: np.array, integration_variable: str,
    ):
        """
        Angular auto convergence power spectrum.
        """
        LinearAngularPowerSpectrumWarning("TODO")

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


