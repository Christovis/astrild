import os
from typing import Dict, List, Optional, Tuple, Type, Union
import subprocess

import numpy as np
import numba as nb
from scipy.integrate import quad
import pandas as pd

from astropy import units as un
from astropy import constants as const
from astropy.cosmology import LambdaCDM
from nbodykit.lab import cosmology
import camb
import lenstools
from lenstools import ConvergenceMap

from wys_ars.rays.skymap import SkyMap
from wys_ars.rays.skys import SkyArray
from wys_ars.rays.skyio import SkyIO
from wys_ars.io import IO

class LinearPowerSpectrumWarning(BaseException):
    pass


class LinearPowerSpectrum:
    """
    Attributes:

    Methods:
    """

    def __init__(
        self,
        astropycosmo: LambdaCDM,
        background_fcts: cosmology.background.PerturbationGrowth,
        Pk_dd: cosmology.LinearPower,
    ):
        self.astropycosmo = astropycosmo
        self.background_fcts = background_fcts
        self.Pk_dd = Pk_dd
        self.H0 = astropycosmo.H0.value
        self.Om0 = astropycosmo.Om0
        self.Ode0 = astropycosmo.Ode0

    @classmethod
    def from_parameters(
        cls,
        H0: float = 67.74,
        Om0: float = 0.3089,
        Ob0: float = 0.0482754208891869,
        Ode0: float = 0.6911,
        T0_cmb: float = 2.7255,
        As: float = 2.105e-9,
        ns: float = 0.9665,
    ) -> "LinearPowerSpectrum":
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
        return cls.from_dictionary(dic)
    
    @classmethod
    def from_dictionary(cls, dic: dict,) -> "LinearPowerSpectrum":
        T0_cmb = dic["T0_cmb"]
        del dic["T0_cmb"], dic["As"], dic["ns"]
        astropycosmo = LambdaCDM(**dic)
        _nbodykitcosmo = cls.init_nbodykit_cosmos_from_astropy_cosmos(
            astropycosmo, T0_cmb, dic["Ob0"],
        )
        # get linear background functions for growth factors/rates
        background_fcts = cosmology.background.PerturbationGrowth(_nbodykitcosmo)
        # P(z=0, k) function of matter density fluctuations [Mpc^3/h^3]
        Pk_dd = cosmology.LinearPower(
            _nbodykitcosmo, redshift=0., transfer='CLASS'
        ) 
        return cls.from_classes(astropycosmo, background_fcts, Pk_dd)
    
    @classmethod
    def from_classes(
        cls,
        astropycosmo: LambdaCDM,
        background_fcts: cosmology.background.PerturbationGrowth,
        Pk_dd: cosmology.LinearPower,
    ) -> "LinearPowerSpectrum":
        return cls(astropycosmo, background_fcts, Pk_dd)

    @staticmethod
    def init_nbodykit_cosmos_from_astropy_cosmos(
        astropycosmo: LambdaCDM, T0_cmb: float = 2.7255, Ob0: float = 0.0482754208891869,
    ) -> cosmology.Cosmology:
        """
        Necessary to obtain P(z=0,k) function of matter density fluctuations
        and background growth rate/factor.
        
        Note: Might replace CAMB, but needs to be tested.
        """
        _dic = cosmology.cosmology.astropy_to_dict(astropycosmo)
        _dic["T0_cmb"] = T0_cmb
        _dic["Omega0_b"] = Ob0
        return cosmology.Cosmology(**_dic)
    
    @staticmethod
    def init_camb_cosmos_from_astropy_cosmos(dic: dict,) -> camb.CAMBparams:
        """
        Necessary to obtain P(z,k) function of matter density fluctuations.
        """
        cambcosmo = camb.CAMBparams()
        cambcosmo.set_cosmology(
            H0=dic["H0"],
            ombh2=dic["Ob0"] * (dic["H0"]/100)**2,
            omch2=(dic["Om0"]-dic["Ob0"]) * (dic["H0"]/100)**2,
            TCMB=dic["T0_cmb"],
        )
        cambcosmo.InitPower.set_params(As=dic["As"], ns=dic["As"])   
        return cambcosmo

    def P_dpdp(
        self, z: float, k: np.array, scale: bool = False,
    ) -> np.array:
        """
        Function of linear power spectrum of the ISW effect P_dpdp(k,t)
        Eq. 6 in arxiv:0809.4488
        to be integrated over time to obtain Cl_tt
        Eq. 2 in arxiv:0809.4488

        Args:
            z:
                Redshift.
            k:
                Wavenumber.
            D:
                Background Growth rate.
            f:
                Background Growth factor.
        """
        _p_dd = self.Pk_dd(k)
        _a = 1/(z + 1)  # scale factor
        D = self.background_fcts.D1(_a)
        f = self.background_fcts.f1(_a)
        
        _E = np.sqrt(self.Om0*(1+z)**3 + self.Ode0)  # astropycosmo.efunc(z)  #H(z)/H0
        _Hz = _E*self.H0
        prefac_dynamic = (_Hz*D*(1-f)/_a)**2
        prefac_static = 9/4 * (self.H0/k)**4 * self.Om0**2
        
        # power spectrum of dt of matter potential fluctuations
        if scale is True:
            print("scaling")
            #p_dpdp = k**3/(2*np.pi**2) * prefac_dynamic * _p_dd
            p_dpdp = (D*(1-f))**2/k**4 * _p_dd
        elif scale == "I don't know":
            p_dpdp = k**3/(2*np.pi**2) * prefac_static * prefac_dynamic * _p_dd
        elif scale is False:
            _H0 = 100 # astropycosmo.H0.value
            _Hz = _E*_H0
            prefac_dynamic = _Hz * (D*(1-f))**2
            p_dpdp = prefac_static * prefac_dynamic * _p_dd

        return p_dpdp
