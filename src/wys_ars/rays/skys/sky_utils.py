import os
from typing import Tuple
import ctypes as ct
from pathlib import Path

import pandas as pd
import numpy as np

c_light = 299792.458  # [km/s]
c_lib_path = Path(__file__).parent.absolute()


class SkyUtils:
    def convert_code_to_phy_units(
        quantity: str, map_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Convert from RayRamses code units to physical units.
        """
        if quantity in ["shear_x", "shear_y", "deflt_x", "deflt_y", "kappa_2"]:
            map_df.loc[:, [quantity]] /= c_light ** 2
        elif quantity in ["isw_rs"]:
            map_df.loc[:, [quantity]] /= c_light ** 3
        return map_df
    
    def convert_convergence_to_deflection(
        kappa: np.ndarray,
        npix: int,
        opening_angle: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Args:
            kappa:
                Convergence map
            opening_angle:
                Edge length of field-of-view [degree]
            npix:
                Number of pixels along edge of field-of-view
        """
        opening_angle *= 3600  # convert degree in arcsec
        alpha1, alpha2 = _call_cal_alphas(kappa, npix, opening_angle)
        return alpha1, alpha2

def _make_r_coor(
    opening_angle: float, npix: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ Returns pixel coordinates """
    ds = opening_angle / npix
    x1edge = np.linspace(0, opening_angle-ds, npix) - opening_angle/2. + ds/2.
    x2, x1 = np.meshgrid(x1edge, x1edge)
    return x1, x2, x1edge

gls = ct.CDLL(c_lib_path / "lib_so_cgls/libglsg.so")
gls.kappa0_to_alphas.argtypes = [
    np.ctypeslib.ndpointer(dtype = ct.c_double),
    ct.c_int,ct.c_double,
    np.ctypeslib.ndpointer(dtype = ct.c_double),
    np.ctypeslib.ndpointer(dtype = ct.c_double),
]
gls.kappa0_to_alphas.restype  = ct.c_void_p
def _call_cal_alphas(
    kappa: np.ndarray, npix: int, opening_angle: float,
) -> Tuple[np.ndarray, np.ndarray]:
    _kappa = np.array(kappa, dtype=ct.c_double)
    alpha1 = np.array(np.zeros((npix,npix)), dtype=ct.c_double)
    alpha2 = np.array(np.zeros((npix,npix)), dtype=ct.c_double)
    gls.kappa0_to_alphas(_kappa, npix, opening_angle, alpha1, alpha2)
    return alpha1, alpha2

gls.kappa0_to_phi.argtypes = [
    np.ctypeslib.ndpointer(dtype = ct.c_double),
    ct.c_int,ct.c_double,
    np.ctypeslib.ndpointer(dtype = ct.c_double),
]
gls.kappa0_to_phi.restype  = ct.c_void_p
def _call_cal_phi(kappa: np.ndarray, npix: int, opening_angle: float) -> np.ndarray:
    _kappa = np.array(kappa, dtype=ct.c_double)
    phi = np.array(np.zeros((npix, npix)), dtype=ct.c_double)
    gls.kappa0_to_phi(_kappa, npix, opening_angle, phi)
    return phi
